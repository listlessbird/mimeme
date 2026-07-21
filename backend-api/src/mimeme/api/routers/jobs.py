from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from mimeme.api.auth import AdminRequired
from mimeme.api.deps import TemporalClientDep
from mimeme.api.models.errors import error_responses
from mimeme.api.models.health import IndexVersionResponse, IndexVersionsResponse
from mimeme.api.models.jobs import (
    IndexFreshnessResponse,
    JobListResponse,
    JobResponse,
    RebuildIndexRequest,
)
from mimeme.db.schema import JobStatus, JobType, RebuildTrigger
from mimeme.domain.job_rules import JobLifecycleInvalidStateError, JobLifecycleNotFoundError
from mimeme.domain.job_store import ApiJobStore
from mimeme.shared.config import settings
from mimeme.workflows import RebuildIndexWorkflow, RebuildIndexWorkflowInput

router = APIRouter(prefix="/jobs", tags=["Jobs"], responses=error_responses(403, 429, 500))


@router.get("/{job_id}", response_model=JobResponse, responses=error_responses(404))
async def get_job(_auth: AdminRequired, job_id: str) -> JobResponse:
    try:
        job = await ApiJobStore().get_job(job_id)
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
    rebuild = await store.create_rebuild_job(
        force=request.force,
        model_name=request.model_name or settings.inference.embed_model,
        index_type=settings.index.type,
    )

    await temporal.start_workflow(
        RebuildIndexWorkflow.run,
        RebuildIndexWorkflowInput(
            job_id=rebuild.job.id,
            force=rebuild.force,
            model_name=rebuild.model_name,
            index_type=rebuild.index_type,
            trigger=RebuildTrigger.MANUAL,
        ),
        id=rebuild.workflow_id,
        task_queue=settings.temporal.task_queue,
    )

    await store.record_workflow_id(rebuild.job.id, rebuild.workflow_id)

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
        cancellation = await store.request_cancellation(job_id)
    except JobLifecycleNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    except JobLifecycleInvalidStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if cancellation.workflow_id:
        handle = temporal.get_workflow_handle(cancellation.workflow_id)
        await handle.cancel()

    await store.mark_cancelled(job_id)
    await store.release_rebuild_claim(job_id)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    _auth: AdminRequired,
    status: Annotated[JobStatus | None, Query()] = None,
    job_type: Annotated[JobType | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> JobListResponse:
    result = await ApiJobStore().list_jobs(status=status, job_type=job_type, limit=limit)

    return JobListResponse(
        jobs=[JobResponse.model_validate(job.model_dump()) for job in result.jobs],
        total=result.total,
    )


@router.get("/indexes/freshness", response_model=IndexFreshnessResponse)
async def get_index_freshness(_auth: AdminRequired) -> IndexFreshnessResponse:
    status = await ApiJobStore().get_index_freshness()
    view = status.view
    return IndexFreshnessResponse(
        desired_generation=view.desired_generation,
        active_generation=view.active_generation,
        is_stale=view.is_stale,
        active_version=status.active_version,
        rebuild_job_id=view.rebuild_job_id,
        rebuild_target_generation=view.rebuild_target_generation,
        rebuild_claimed_at=view.rebuild_claimed_at,
        last_dirty_at=view.last_dirty_at,
        last_dirty_reason=view.last_dirty_reason,
        last_reconciled_at=view.last_reconciled_at,
    )


@router.get("/indexes/versions", response_model=IndexVersionsResponse)
async def list_index_versions(
    _auth: AdminRequired,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> IndexVersionsResponse:
    builds = await ApiJobStore().list_index_builds(limit=limit)

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
