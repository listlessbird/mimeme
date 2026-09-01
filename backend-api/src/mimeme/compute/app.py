from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from mimeme import search, storage
from mimeme.compute.jobs import Jobs
from mimeme.compute.model import (
    ChildErr,
    ChildOk,
    ImageInfo,
    InspectCall,
    JobSpec,
    JobState,
    Readiness,
    RoleStatus,
    StorageRole,
)
from mimeme.compute.supervisor import ChildDead, Supervisor
from mimeme.compute.workspace import Workspace
from mimeme.config import ArtifactConfig, MediaConfig, Settings
from mimeme.logging import setup_logging
from mimeme.search.gateway import Gateway as SearchGateway

_MAX_IMAGE_BYTES = 64 * 1024 * 1024


class InspectRequest(BaseModel):
    key: str
    role: StorageRole = "media"


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
        supervisor = Supervisor(
            socket_dir,
            call_timeout_s=settings.compute.request_timeout_s,
        )
        await supervisor.start()
        jobs = Jobs(
            supervisor,
            media=media,
            artifacts=artifacts,
            workspace_dir=workspace_dir,
            io_concurrency=settings.compute.job_io_concurrency,
            residency_mode=settings.inference.residency,
        )
        app.state.media = media
        app.state.artifacts = artifacts
        app.state.supervisor = supervisor
        app.state.jobs = jobs
        app.state.search = SearchGateway(
            supervisor,
            artifacts=artifacts,
            workspace_dir=workspace_dir,
        )
        try:
            yield
        finally:
            await supervisor.close()
            await artifacts.close()
            await media.close()

    app = FastAPI(title="mimeme-compute", lifespan=lifespan)

    @app.get("/ready", response_model=Readiness)
    async def ready() -> Readiness:
        readiness = app.state.supervisor.readiness()
        gateway: SearchGateway = app.state.search
        try:
            status = await gateway.status()
            search_role = RoleStatus(
                role="search",
                state="ready" if status.ready else "starting",
                detail=status.detail,
                loaded_version=status.serving_version,
                model_identity=(
                    f"{status.embed_model}:{status.encoder_repo}"
                    f"@{status.encoder_revision}/{status.encoder_variant}"
                    if status.embed_model
                    and status.encoder_repo
                    and status.encoder_revision
                    and status.encoder_variant
                    else status.embed_model
                ),
            )
        except search.Error as exc:
            search_role = RoleStatus(role="search", state="failed", detail=str(exc))
            status = None
        roles = [search_role if role.role == "search" else role for role in readiness.roles]
        return Readiness(ok=readiness.ok and status is not None and status.ready, roles=roles)

    @app.get("/v1/roles/inference/ready", response_model=Readiness)
    async def inference_ready() -> Readiness:
        role = app.state.supervisor.role_status("inference")
        return Readiness(ok=role.state == "ready", roles=[role])

    @app.get("/v1/search/status", response_model=search.Status)
    async def search_status() -> search.Status:
        gateway: SearchGateway = app.state.search
        try:
            return await gateway.status()
        except search.Error as exc:
            raise _search_http(exc) from exc

    @app.post("/v1/search/query", response_model=search.Batch)
    async def search_query(request: search.CandidateRequest) -> search.Batch:
        gateway: SearchGateway = app.state.search
        try:
            return await gateway.query(request)
        except search.Error as exc:
            raise _search_http(exc) from exc

    @app.post("/v1/search/load", response_model=search.Loaded)
    async def search_load(request: search.Load) -> search.Loaded:
        gateway: SearchGateway = app.state.search
        try:
            return await gateway.load(request)
        except search.Error as exc:
            raise _search_http(exc) from exc

    @app.post("/v1/search/switch", response_model=search.Status)
    async def search_switch(request: search.Switch) -> search.Status:
        gateway: SearchGateway = app.state.search
        try:
            return await gateway.switch(request.version)
        except search.Error as exc:
            raise _search_http(exc) from exc

    @app.post("/v1/search/rollback", response_model=search.Status)
    async def search_rollback(request: search.Rollback) -> search.Status:
        gateway: SearchGateway = app.state.search
        try:
            return await gateway.rollback(request.failed_version)
        except search.Error as exc:
            raise _search_http(exc) from exc

    @app.post("/v1/search/clear", response_model=search.Status)
    async def search_clear() -> search.Status:
        gateway: SearchGateway = app.state.search
        try:
            return await gateway.clear()
        except search.Error as exc:
            raise _search_http(exc) from exc

    @app.post("/v1/image/inspect", response_model=ImageInfo)
    async def image_inspect(request: InspectRequest) -> ImageInfo:
        supervisor: Supervisor = app.state.supervisor
        store: storage.Store = (
            app.state.artifacts if request.role == "artifacts" else app.state.media
        )
        workspace = Workspace.create(workspace_dir, "inspect")
        try:
            data = await store.read_bytes(storage.Object(request.key), max_bytes=_MAX_IMAGE_BYTES)
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
    async def get_job(job_id: str, wait_s: Annotated[float, Query(ge=0, le=30)] = 0) -> JobState:
        jobs: Jobs = app.state.jobs
        state = await jobs.wait(job_id, wait_s) if wait_s > 0 else jobs.get(job_id)
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


def _search_http(exc: search.Error) -> HTTPException:
    if isinstance(exc, (search.Unavailable, search.Loading)):
        status_code = 503
    elif isinstance(exc, search.NotFound):
        status_code = 404
    elif isinstance(exc, search.Invalid):
        status_code = 400
    elif isinstance(exc, (search.Incompatible, search.Stale)):
        status_code = 409
    else:
        status_code = 500
    return HTTPException(
        status_code=status_code,
        detail={"code": exc.code, "message": str(exc)},
    )


def run() -> None:
    import uvicorn

    settings = Settings()
    setup_logging(settings, "compute")
    host, port = _bind(settings)
    uvicorn.run(create_app(settings), host=host, port=port)


def _bind(settings: Settings) -> tuple[str, int]:
    return settings.compute.bind_host, settings.compute.bind_port


if __name__ == "__main__":
    run()
