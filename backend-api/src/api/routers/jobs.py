from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from api.auth import AdminRequired
from api.deps import DbSession, TemporalClientDep
from api.models.errors import error_responses
from api.models.health import IndexVersionResponse, IndexVersionsResponse
from api.models.jobs import JobListResponse, JobResponse, RebuildIndexRequest
from domain.job_store import (
    ApiJobStore,
    JobLifecycleInvalidStateError,
    JobLifecycleNotFoundError,
)
from shared.config import settings
from shared.models import IndexBuild, JobStatus, JobType
from workflows import RebuildIndexWorkflow, RebuildIndexWorkflowInput

router = APIRouter(prefix="/jobs", tags=["Jobs"], responses=error_responses(403, 429, 500))


@router.get("/{job_id}", response_model=JobResponse, responses=error_responses(404))
async def get_job(_auth: AdminRequired, job_id: str) -> JobResponse:
    try:
        job = ApiJobStore().get_job(job_id)
    except JobLifecycleNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse.model_validate(job.model_dump())


@router.post("/rebuild-index", response_model=JobResponse, status_code=202)
async def trigger_rebuild_index(
    _auth: AdminRequired,
    temporal: TemporalClientDep,
    request: RebuildIndexRequest | None = None,
) -> JobResponse:
    request = request or RebuildIndexRequest()
    store = ApiJobStore()
    rebuild = store.create_rebuild_job(
        force=request.force,
        model_name=request.model_name or settings.embed_model,
        index_type=settings.index_type,
    )

    await temporal.start_workflow(
        RebuildIndexWorkflow.run,
        RebuildIndexWorkflowInput(
            job_id=rebuild.job.id,
            force=rebuild.force,
            model_name=rebuild.model_name,
            index_type=rebuild.index_type,
        ),
        id=rebuild.workflow_id,
        task_queue=settings.temporal_task_queue,
    )

    store.record_workflow_id(rebuild.job.id, rebuild.workflow_id)

    return JobResponse(
        id=rebuild.job.id,
        type=JobType.REBUILD_INDEX,
        status=JobStatus.PENDING,
        progress=0.0,
        message="Index rebuild queued",
        created_at=rebuild.job.created_at,
    )


@router.delete("/{job_id}", status_code=204, responses=error_responses(400, 404))
async def cancel_job(_auth: AdminRequired, job_id: str, temporal: TemporalClientDep) -> None:
    store = ApiJobStore()
    try:
        cancellation = store.request_cancellation(job_id)
    except JobLifecycleNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    except JobLifecycleInvalidStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if cancellation.workflow_id:
        handle = temporal.get_workflow_handle(cancellation.workflow_id)
        await handle.cancel()

    store.mark_cancelled(job_id)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    _auth: AdminRequired,
    status: Annotated[JobStatus | None, Query()] = None,
    job_type: Annotated[JobType | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> JobListResponse:
    result = ApiJobStore().list_jobs(status=status, job_type=job_type, limit=limit)

    return JobListResponse(
        jobs=[JobResponse.model_validate(job.model_dump()) for job in result.jobs],
        total=result.total,
    )


@router.get("/indexes/versions", response_model=IndexVersionsResponse)
async def list_index_versions(
    _auth: AdminRequired,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> IndexVersionsResponse:

    builds = db.scalars(
        select(IndexBuild).order_by(IndexBuild.created_at.desc()).limit(limit)
    ).all()

    return IndexVersionsResponse(
        versions=[
            IndexVersionResponse(
                version=b.version,
                embed_model=b.embed_model,
                index_type=b.index_type,
                num_vectors=b.num_vectors,
                dimension=b.dimension,
                is_active=b.is_active,
                created_at=b.created_at.isoformat() if b.created_at else None,
            )
            for b in builds
        ]
    )
