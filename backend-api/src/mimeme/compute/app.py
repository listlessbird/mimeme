from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mimeme import storage
from mimeme.compute.jobs import Jobs
from mimeme.compute.model import (
    ChildErr,
    ChildOk,
    ImageInfo,
    InspectCall,
    JobSpec,
    JobState,
    Readiness,
)
from mimeme.compute.supervisor import ChildDead, Supervisor
from mimeme.compute.workspace import Workspace
from mimeme.shared.config import ArtifactConfig, MediaConfig, Settings

_MAX_IMAGE_BYTES = 64 * 1024 * 1024


class InspectRequest(BaseModel):
    key: str


def _storage_config(config: MediaConfig | ArtifactConfig) -> storage.Config:
    return storage.Config(
        endpoint_url=config.s3_endpoint_url,
        region=config.s3_region,
        access_key=config.s3_access_key_id,
        secret_key=config.s3_secret_access_key,
        bucket=config.s3_bucket,
        force_path_style=config.s3_force_path_style,
    )


def create_app(settings: Settings) -> FastAPI:
    socket_dir = settings.compute.socket_dir
    workspace_dir = socket_dir / "work"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        media = await storage.S3.open(_storage_config(settings.media))
        artifacts = await storage.S3.open(_storage_config(settings.artifacts))
        supervisor = Supervisor(socket_dir)
        await supervisor.start()
        jobs = Jobs(
            supervisor,
            media=media,
            artifacts=artifacts,
            workspace_dir=workspace_dir,
        )
        app.state.media = media
        app.state.artifacts = artifacts
        app.state.supervisor = supervisor
        app.state.jobs = jobs
        try:
            yield
        finally:
            await supervisor.close()
            await artifacts.close()
            await media.close()

    app = FastAPI(title="mimeme-compute", lifespan=lifespan)

    @app.get("/ready", response_model=Readiness)
    async def ready() -> Readiness:
        return app.state.supervisor.readiness()

    @app.post("/v1/image/inspect", response_model=ImageInfo)
    async def image_inspect(request: InspectRequest) -> ImageInfo:
        supervisor: Supervisor = app.state.supervisor
        media: storage.Store = app.state.media
        workspace = Workspace.create(workspace_dir, "inspect")
        try:
            data = await media.read_bytes(storage.Object(request.key), max_bytes=_MAX_IMAGE_BYTES)
            path = workspace.write_atomic("input", data)
            call = InspectCall(path=str(path))
            try:
                raw = await supervisor.call("image", call.model_dump_json().encode("utf-8"))
            except ChildDead as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            response = _parse_child(raw)
            if isinstance(response, ChildErr):
                raise HTTPException(status_code=422, detail=response.error)
            return ImageInfo.model_validate(response.result)
        except storage.Missing as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        finally:
            workspace.close()

    @app.put("/v1/jobs/{job_id}", response_model=JobState)
    async def submit_job(job_id: str, spec: JobSpec) -> JobState:
        jobs: Jobs = app.state.jobs
        return jobs.submit(job_id, spec)

    @app.get("/v1/jobs/{job_id}", response_model=JobState)
    async def get_job(job_id: str) -> JobState:
        jobs: Jobs = app.state.jobs
        state = jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="job not found")
        return state

    @app.delete("/v1/jobs/{job_id}", response_model=JobState)
    async def cancel_job(job_id: str) -> JobState:
        jobs: Jobs = app.state.jobs
        state = await jobs.cancel(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="job not found")
        return state

    return app


def _parse_child(raw: bytes) -> ChildOk | ChildErr:
    import json

    payload = json.loads(raw)
    if payload.get("ok"):
        return ChildOk.model_validate(payload)
    return ChildErr.model_validate(payload)


def run() -> None:
    import uvicorn

    settings = Settings()
    host, port = _bind(settings)
    uvicorn.run(create_app(settings), host=host, port=port)


def _bind(settings: Settings) -> tuple[str, int]:
    return settings.compute.bind_host, settings.compute.bind_port


if __name__ == "__main__":
    run()
