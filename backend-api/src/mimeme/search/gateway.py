from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Protocol, TypeVar

import anyio
import anyio.to_thread
from pydantic import BaseModel, TypeAdapter, ValidationError

from mimeme import storage
from mimeme.compute.model import ChildErr, ChildResponse, Role
from mimeme.compute.supervisor import ChildDead
from mimeme.search import generation_workspace
from mimeme.search.error import (
    Error,
    Failed,
    Incompatible,
    Invalid,
    Loading,
    NotFound,
    Stale,
    Unavailable,
)
from mimeme.search.model import (
    Batch,
    Bm25File,
    CandidateRequest,
    ClearCall,
    File,
    Load,
    LoadCall,
    Loaded,
    PreparedLoad,
    QueryCall,
    RollbackCall,
    Status,
    StatusCall,
    SwitchCall,
)

_T = TypeVar("_T", bound=BaseModel)
_ERRORS: dict[str, type[Error]] = {
    error.code: error
    for error in (Unavailable, Loading, Incompatible, Invalid, NotFound, Stale, Failed)
}
_CHILD_RESPONSE = TypeAdapter(ChildResponse)


class Calls(Protocol):
    async def call(self, role: Role, request: bytes) -> bytes: ...

    async def restart(self, role: Role) -> None: ...


class Gateway:
    def __init__(
        self,
        supervisor: Calls,
        *,
        artifacts: storage.Store,
        workspace_dir: Path,
    ) -> None:
        self._supervisor = supervisor
        self._artifacts = artifacts
        self._workspace_dir = workspace_dir
        self._generations: dict[str, Load] = {}
        self._serving_version: str | None = None
        self._retained_version: str | None = None
        self._recovery_lock = asyncio.Lock()
        self._workspaces: set[Path] = set()

    async def query(self, request: CandidateRequest) -> Batch:
        return await self._call(QueryCall(request=request), Batch)

    async def status(self) -> Status:
        status = await self._call(StatusCall(), Status)
        self._remember_status(status)
        return status

    async def load(self, generation: Load) -> Loaded:
        loaded = await self._load_generation(generation)
        self._generations[generation.version] = generation
        return loaded

    async def _load_generation(self, generation: Load, *, recover: bool = True) -> Loaded:
        root = await anyio.to_thread.run_sync(
            generation_workspace.prepare, self._workspace_dir, generation.version
        )
        self._workspaces.add(root)
        paths: dict[str, str] = {}
        loaded = False
        try:
            for artifact in generation.files:
                paths[artifact.name] = str(await self._download(root, artifact))
            if generation.bm25 is not None:
                paths[generation.bm25.name] = str(await self._download(root, generation.bm25))
            call = PreparedLoad(
                version=generation.version,
                workspace=str(root),
                paths=paths,
                bm25=generation.bm25,
                encoder=generation.encoder,
                hnsw_ef_search=generation.hnsw_ef_search,
            )
            result = await self._call(LoadCall(load=call), Loaded, recover=recover)
            loaded = True
            return result
        finally:
            if not loaded:
                await anyio.to_thread.run_sync(generation_workspace.discard, root)
                self._workspaces.discard(root)

    async def switch(self, version: str) -> Status:
        status = await self._call(SwitchCall(version=version), Status)
        self._remember_status(status)
        return status

    async def rollback(self, failed_version: str) -> Status:
        status = await self._call(RollbackCall(failed_version=failed_version), Status)
        self._remember_status(status)
        return status

    async def clear(self) -> Status:
        async with self._recovery_lock:
            try:
                await self._call(ClearCall(), Status, recover=False)
            except Error:
                pass
            await self._supervisor.restart("search")
            await self._discard_workspaces()
            self._generations.clear()
            self._serving_version = None
            self._retained_version = None
            status = await self._call(StatusCall(), Status, recover=False)
            self._remember_status(status)
            return status

    async def _download(self, root: Path, artifact: File | Bm25File) -> Path:
        target = root / artifact.name
        digest = hashlib.sha256()
        try:
            async with self._artifacts.read(storage.Object(artifact.key)) as chunks:
                async with await anyio.open_file(target, "wb") as handle:
                    async for chunk in chunks:
                        digest.update(chunk)
                        await handle.write(chunk)
        except storage.Missing as exc:
            raise Incompatible(f"search artifact is missing: {artifact.key}") from exc
        except storage.Error as exc:
            raise Unavailable(f"search artifact read failed: {artifact.key}: {exc}") from exc
        if digest.hexdigest() != artifact.sha256:
            await anyio.Path(target).unlink(missing_ok=True)
            raise Incompatible(f"search artifact checksum mismatch: {artifact.key}")
        return target

    async def _call(
        self,
        call: BaseModel,
        model: type[_T],
        *,
        recover: bool = True,
    ) -> _T:
        try:
            raw = await self._supervisor.call("search", call.model_dump_json().encode("utf-8"))
        except ChildDead as exc:
            if not recover:
                raise Unavailable("search child failed during recovery") from exc
            await self._recover(exc)
            raise Unavailable(
                "search child restarted and restored its serving generation; retry the request"
            ) from exc
        try:
            response = _CHILD_RESPONSE.validate_json(raw)
        except (ValueError, ValidationError) as exc:
            raise Failed(f"invalid response from search child: {exc}") from exc
        if isinstance(response, ChildErr):
            raise _ERRORS.get(response.code or "search_failed", Failed)(response.error)
        try:
            return model.model_validate(response.result)
        except ValidationError as exc:
            raise Failed(f"invalid search child result: {exc}") from exc

    def _remember_status(self, status: Status) -> None:
        self._serving_version = status.serving_version
        self._retained_version = status.retained_version

    async def _recover(self, cause: ChildDead) -> None:
        async with self._recovery_lock:
            try:
                try:
                    current = await self._call(StatusCall(), Status, recover=False)
                except Error:
                    current = None
                if (
                    current is not None
                    and self._serving_version is not None
                    and current.serving_version == self._serving_version
                    and current.retained_version == self._retained_version
                ):
                    self._remember_status(current)
                    return

                await self._supervisor.restart("search")
                await self._discard_workspaces()
                retained = self._known_generation(self._retained_version)
                serving = self._known_generation(self._serving_version)
                if self._retained_version is not None and retained is None:
                    raise Unavailable(
                        "search child restarted without its known rollback generation"
                    )
                if serving is None:
                    raise Unavailable("search child restarted without a known serving generation")
                if retained is not None:
                    await self._load_generation(retained, recover=False)
                    await self._call(SwitchCall(version=retained.version), Status, recover=False)
                await self._load_generation(serving, recover=False)
                status = await self._call(
                    SwitchCall(version=serving.version), Status, recover=False
                )
                self._remember_status(status)
            except Error:
                raise
            except Exception as exc:
                raise Unavailable(f"search child recovery failed: {exc}") from cause

    def _known_generation(self, version: str | None) -> Load | None:
        return self._generations.get(version) if version else None

    async def _discard_workspaces(self) -> None:
        workspaces = tuple(self._workspaces)
        for root in workspaces:
            await anyio.to_thread.run_sync(generation_workspace.discard, root)
        self._workspaces.clear()
