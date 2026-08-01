from __future__ import annotations

import asyncio
from pathlib import Path

from mimeme import storage
from mimeme.compute.model import (
    AnnotateCall,
    AnnotateResult,
    AnnotateSpec,
    ChildErr,
    ChildOk,
    EmbedCall,
    EmbedCallItem,
    EmbedReply,
    EmbedResult,
    EmbedResultItem,
    EmbedSpec,
    JobSpec,
    JobState,
)
from mimeme.compute.supervisor import ChildDead, Supervisor
from mimeme.compute.workspace import Workspace
from mimeme.index.gateway import Gateway as IndexGateway
from mimeme.index.model import BuildSpec

_MAX_IMAGE_BYTES = 64 * 1024 * 1024


class ComputeError(Exception):
    pass


class _Job:
    def __init__(self, state: JobState, spec: JobSpec) -> None:
        self.state = state
        self.spec = spec
        self.task: asyncio.Task[None] | None = None
        self.running = False


class Jobs:
    def __init__(
        self,
        supervisor: Supervisor,
        *,
        media: storage.Store,
        artifacts: storage.Store,
        workspace_dir: Path,
    ) -> None:
        self._supervisor = supervisor
        self._media = media
        self._artifacts = artifacts
        self._workspace_dir = workspace_dir
        self._jobs: dict[str, _Job] = {}
        self._index = IndexGateway(
            supervisor,
            artifacts=artifacts,
            workspace_dir=workspace_dir,
        )

    def get(self, job_id: str) -> JobState | None:
        job = self._jobs.get(job_id)
        return job.state.model_copy(deep=True) if job else None

    def submit(self, job_id: str, spec: JobSpec) -> JobState:
        existing = self._jobs.get(job_id)
        if existing is not None:
            if existing.spec != spec:
                raise ComputeError(f"job {job_id} already exists with a different request")
            if existing.state.status != "failed":
                return existing.state.model_copy(deep=True)
        job = _Job(JobState(job_id=job_id, status="queued"), spec)
        self._jobs[job_id] = job
        job.task = asyncio.create_task(self._run(job))
        return job.state.model_copy(deep=True)

    async def cancel(self, job_id: str) -> JobState | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.state.status in ("succeeded", "failed", "cancelled"):
            return job.state.model_copy(deep=True)
        was_running = job.running
        job.state.status = "cancelled"
        job.state.error = "cancelled"
        if job.task is not None:
            job.task.cancel()
        if was_running:
            await self._supervisor.restart(
                "index" if isinstance(job.spec, BuildSpec) else "inference"
            )
        return job.state.model_copy(deep=True)

    async def _run(self, job: _Job) -> None:
        workspace = Workspace.create(self._workspace_dir, job.state.job_id)
        try:
            job.state.status = "running"
            job.running = True
            spec = job.spec
            if isinstance(spec, AnnotateSpec):
                result = await self._run_annotate(job, spec, workspace)
            elif isinstance(spec, EmbedSpec):
                result = await self._run_embed(job, spec, workspace)
            else:
                result = await self._run_index(job, spec)
            if job.state.status == "cancelled":
                return
            job.state.status = "succeeded"
            job.state.progress = 1.0
            job.state.phase = "done"
            job.state.result = result
        except asyncio.CancelledError:
            job.state.status = "cancelled"
            job.state.error = "cancelled"
            raise
        except ChildDead as exc:
            job.state.status = "failed"
            job.state.error = f"child_dead: {exc}"
        except Exception as exc:
            job.state.status = "failed"
            job.state.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.running = False
            workspace.close()

    async def _run_annotate(self, job: _Job, spec: AnnotateSpec, workspace: Workspace) -> dict:
        job.state.phase = "download"
        data = await self._media.read_bytes(
            storage.Object(spec.media_key), max_bytes=_MAX_IMAGE_BYTES
        )
        path = workspace.write_atomic("input", data)
        job.state.phase = "annotate"
        job.state.progress = 0.5
        call = AnnotateCall(path=str(path), length=spec.length)
        reply = await self._call_inference(call.model_dump_json().encode("utf-8"))
        result = AnnotateResult.model_validate(reply.result)
        return result.model_dump()

    async def _run_embed(self, job: _Job, spec: EmbedSpec, workspace: Workspace) -> dict:
        job.state.phase = "download"
        call_items: list[EmbedCallItem] = []
        for index, item in enumerate(spec.items):
            data = await self._media.read_bytes(
                storage.Object(item.media_key), max_bytes=_MAX_IMAGE_BYTES
            )
            path = workspace.write_atomic(f"in_{index}", data)
            call_items.append(
                EmbedCallItem(
                    image_id=item.image_id,
                    path=str(path),
                    text=item.text,
                    image_out=str(workspace.path(f"img_{index}.npy")),
                    text_out=str(workspace.path(f"txt_{index}.npy")),
                )
            )

        job.state.phase = "embed"
        job.state.progress = 0.5
        call = EmbedCall(items=call_items)
        reply = await self._call_inference(call.model_dump_json().encode("utf-8"))
        embed_reply = EmbedReply.model_validate(reply.result)

        job.state.phase = "upload"
        job.state.progress = 0.8
        by_id = {item.image_id: item for item in spec.items}
        call_by_id = {item.image_id: item for item in call_items}
        results: list[EmbedResultItem] = []
        for reply_item in embed_reply.items:
            spec_item = by_id[reply_item.image_id]
            if not reply_item.ok:
                results.append(
                    EmbedResultItem(image_id=reply_item.image_id, ok=False, error=reply_item.error)
                )
                continue
            call_item = call_by_id[reply_item.image_id]
            image_bytes = Path(call_item.image_out).read_bytes()
            text_bytes = Path(call_item.text_out).read_bytes()
            await self._artifacts.put_bytes(
                storage.Object(spec_item.image_key),
                image_bytes,
                content_type="application/octet-stream",
            )
            await self._artifacts.put_bytes(
                storage.Object(spec_item.text_key),
                text_bytes,
                content_type="application/octet-stream",
            )
            results.append(
                EmbedResultItem(
                    image_id=reply_item.image_id,
                    ok=True,
                    image_key=spec_item.image_key,
                    text_key=spec_item.text_key,
                    model=reply_item.model,
                    dimension=reply_item.dimension,
                )
            )
        return EmbedResult(items=results).model_dump()

    async def _run_index(self, job: _Job, spec: BuildSpec) -> dict:
        async def progress(phase: str, value: float) -> None:
            job.state.phase = phase
            job.state.progress = value

        result = await self._index.build(spec.build, progress=progress)
        return result.model_dump()

    async def _call_inference(self, request: bytes) -> ChildOk:
        raw = await self._supervisor.call("inference", request)
        response = _parse_child(raw)
        if isinstance(response, ChildErr):
            raise ComputeError(response.error)
        return response


def _parse_child(raw: bytes) -> ChildOk | ChildErr:
    import json

    payload = json.loads(raw)
    if payload.get("ok"):
        return ChildOk.model_validate(payload)
    return ChildErr.model_validate(payload)
