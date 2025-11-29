import uuid
from datetime import datetime

import structlog
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query

from api.models.jobs import JobListResponse, JobResponse, JobStatus, RebuildIndexRequest
from api.tasks import celery_app
from api.tasks.manage_index import rebuild_index_task

log = structlog.get_logger()
router = APIRouter()


def _celery_state_to_status(state: str) -> JobStatus:
    mapping = {
        "PENDING": JobStatus.PENDING,
        "STARTED": JobStatus.RUNNING,
        "PROGRESS": JobStatus.RUNNING,
        "SUCCESS": JobStatus.SUCCESS,
        "FAILURE": JobStatus.FAILED,
        "REVOKED": JobStatus.CANCELLED,
    }
    return mapping.get(state, JobStatus.PENDING)


@router.get("/{job_id}", response_model=JobResponse)
async def get_jon(job_id: str) -> JobResponse:
    result = AsyncResult(job_id, app=celery_app)

    info = result.info or {}

    job_type = "unknown"
    if job_id.startswith("ingest-"):
        job_type = "ingest"
    elif job_id.startswith("rebuild-"):
        job_type = "rebuild_index"
    elif job_id.startswith("annotate-"):
        job_type = "annotate"
    elif job_id.startswith("embed-"):
        job_type = "embed"

    status = _celery_state_to_status(result.state)

    progress = 0.0
    message = None

    if isinstance(info, dict):
        progress = info.get("progress", 0.0)
        message = None
    elif result.state == "FAILURE":
        message = str(result.result) if result.result else "Task Failed"

    created_at = datetime.utcnow()
    started_at = None
    completed_at = None

    if result.state in ("SUCCESS", "FAILURE"):
        completed_at = datetime.utcnow()
    if result.state not in ("PENDING",):
        started_at = datetime.utcnow()

    result_data = None
    if result.state == "SUCCESS" and result.result:
        result_data = result.result if isinstance(result.result, dict) else {"value": result.result}

    return JobResponse(
        id=job_id,
        type=job_type,
        status=status,
        progress=progress,
        message=message,
        created_at=created_at,
        started_at=started_at,
        completed_at=completed_at,
        result=result_data,
    )


@router.post("/rebuild-index", response_model=JobResponse, status_code=202)
async def trigger_rebuild_index(request: RebuildIndexRequest | None = None) -> JobResponse:
    request = request or RebuildIndexRequest()
    job_id = f"rebuild-{uuid.uuid4().hex[:12]}"

    rebuild_index_task.apply_async(
        kwargs={
            "force": request.force,
            "model_name": request.model_name,
        },
        task_id=job_id,
    )

    log.info("index_rebuild_queued", job_id=job_id, force=request.force)

    return JobResponse(
        id=job_id,
        type="rebuild_index",
        status=JobStatus.PENDING,
        progress=0.0,
        message="Index rebuild queued",
        created_at=datetime.utcnow(),
    )


@router.delete("/{job_id}", status_code=204)
async def cancel_job(job_id: str) -> None:
    result = AsyncResult(job_id, app=celery_app)

    if result.state in ("SUCCESS", "FAILURE"):
        raise HTTPException(
            status_code=400,
            detail="Cannot cancel completed job",
        )

    result.revoke(terminate=True)
    log.info("job_cancelled", job_id=job_id)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: JobStatus | None = Query(default=None),
    job_type: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> JobListResponse:
    # todo: setup a dedicated job tracking table
    return JobListResponse(
        jobs=[],
        total=0,
    )
