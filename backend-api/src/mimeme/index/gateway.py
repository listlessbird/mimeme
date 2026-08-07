from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import anyio
import structlog
from pydantic import ValidationError

from mimeme import storage
from mimeme.compute.model import ChildErr, ChildOk, Role
from mimeme.compute.workspace import Workspace
from mimeme.index import pack
from mimeme.index.client import Progress
from mimeme.index.model import (
    Build,
    BuildCall,
    Built,
    File,
    LocalEmbedding,
    LocalShard,
    Manifest,
    PreparedBuild,
    Result,
)

log = structlog.getLogger()


class Failed(Exception):
    pass


class Calls(Protocol):
    async def call(self, role: Role, request: bytes) -> bytes: ...


class Gateway:
    def __init__(self, calls: Calls, *, artifacts: storage.Store, workspace_dir: Path) -> None:
        self._calls = calls
        self._artifacts = artifacts
        self._workspace_dir = workspace_dir

    async def build(self, request: Build, *, progress: Progress | None = None) -> Result:
        if not request.embeddings and request.dimension == 0:
            return Result(outcome="empty")
        meter = storage.Meter(self._artifacts)
        workspace = Workspace.create(self._workspace_dir, f"index-{request.version}")
        try:
            if progress is not None:
                await progress("download", 0.1)
            shards = await _fetch_shards(meter, request, workspace)
            local: list[LocalEmbedding] = []
            for position, item in enumerate(request.embeddings):
                if item.sealed:
                    local.append(
                        LocalEmbedding(
                            image_id=item.image_id,
                            shard=item.shard,
                            row=item.row,
                            text_present=item.text_present,
                        )
                    )
                    continue
                assert item.image_key is not None
                image_path = workspace.path(f"input-{position}.npy")
                await _download(meter, item.image_key, image_path)
                text_path: Path | None = None
                if item.text_key is not None:
                    text_path = workspace.path(f"text-{position}.npy")
                    await _download(meter, item.text_key, text_path)
                local.append(
                    LocalEmbedding(
                        image_id=item.image_id,
                        image_path=str(image_path),
                        text_path=str(text_path) if text_path else None,
                    )
                )
            output = workspace.path("output")
            output.mkdir()
            if progress is not None:
                await progress("build", 0.5)
            raw = await self._calls.call(
                "index",
                BuildCall(
                    build=PreparedBuild(
                        version=request.version,
                        target_generation=request.target_generation,
                        model=request.model,
                        index_type=request.index_type,
                        dimension=request.dimension,
                        native_threads=request.native_threads,
                        encoder=request.encoder,
                        output_dir=str(output),
                        shards=shards,
                        embeddings=local,
                    )
                )
                .model_dump_json()
                .encode(),
            )
            response = _parse_child(raw)
            if isinstance(response, ChildErr):
                raise Failed(response.error)
            built = Built.model_validate(response.result)
            _match(request, built)

            if progress is not None:
                await progress("upload", 0.8)
            files: list[File] = []
            for artifact in built.files:
                key = f"indexes/{request.version}/{artifact.name}"
                await _upload(meter, artifact.path, key, artifact.length, artifact.sha256)
                files.append(
                    File(
                        name=artifact.name,
                        key=key,
                        length=artifact.length,
                        sha256=artifact.sha256,
                    )
                )
            manifest = Manifest(
                version=built.version,
                target_generation=built.target_generation,
                model=built.model,
                index_type=built.index_type,
                encoder=request.encoder,
                dimension=built.dimension,
                image_count=built.image_count,
                text_count=built.text_count,
                files=files,
                complete_key=f"indexes/{built.version}/complete.json",
            )
            await meter.put_bytes(
                storage.Object(manifest.complete_key),
                manifest.model_dump_json().encode(),
                content_type="application/json",
            )
            return Result(outcome="built", manifest=manifest)
        finally:
            workspace.close()
            counts = meter.counts
            log.info(
                "index.rebuild.storage",
                job_id=request.job_id,
                version=request.version,
                target_generation=request.target_generation,
                images=len(request.embeddings),
                planned_reads=request.planned_reads,
                class_a=counts.class_a,
                class_b=counts.class_b,
                **counts.model_dump(),
            )


async def _fetch_shards(
    artifacts: storage.Store, request: Build, workspace: Workspace
) -> list[LocalShard]:
    wanted: dict[int, bool] = {}
    for item in request.embeddings:
        if item.shard is None:
            continue
        wanted[item.shard] = wanted.get(item.shard, False) or item.text_present
    shards: list[LocalShard] = []
    for number in sorted(wanted):
        image_path = workspace.path(f"shard-image-{number:06d}.npy")
        await _download(artifacts, pack.locate(request.model, number, text=False), image_path)
        text_path: Path | None = None
        if wanted[number]:
            text_path = workspace.path(f"shard-text-{number:06d}.npy")
            await _download(artifacts, pack.locate(request.model, number, text=True), text_path)
        shards.append(
            LocalShard(
                number=number,
                image_path=str(image_path),
                text_path=str(text_path) if text_path else None,
            )
        )
    return shards


async def _download(artifacts: storage.Store, key: str, target: Path) -> None:
    async with artifacts.read(storage.Object(key)) as chunks:
        async with await anyio.open_file(target, "wb") as handle:
            async for chunk in chunks:
                await handle.write(chunk)


async def _upload(artifacts: storage.Store, path: str, key: str, length: int, sha256: str) -> None:
    async def chunks():  # noqa: ANN202
        async with await anyio.open_file(path, "rb") as handle:
            while chunk := await handle.read(1024 * 1024):
                yield chunk

    await artifacts.put(
        storage.Object(key),
        chunks(),
        length=length,
        content_type="application/octet-stream",
        checksum=storage.Checksum(value=sha256),
    )


def _parse_child(raw: bytes) -> ChildOk | ChildErr:
    try:
        payload = json.loads(raw)
        return (
            ChildOk.model_validate(payload)
            if payload.get("ok")
            else ChildErr.model_validate(payload)
        )
    except (ValueError, ValidationError) as exc:
        raise Failed(f"invalid index child response: {exc}") from exc


def _match(request: Build, built: Built) -> None:
    expected = (request.version, request.target_generation, request.model, request.index_type)
    actual = (built.version, built.target_generation, built.model, built.index_type)
    if actual != expected:
        raise Failed("index child result does not match the requested generation")
