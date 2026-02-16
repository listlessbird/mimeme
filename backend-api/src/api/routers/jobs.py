from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException, Query

from api.auth import AdminRequired
from api.deps import DbSession, TemporalClientDep
from api.models.jobs import JobListResponse, JobResponse, RebuildIndexRequest
from shared.config import settings
from shared.models import IndexBuild, Job, JobStatus, JobType
from workflows import RebuildIndexWorkflow, RebuildIndexWorkflowInput

router = APIRouter()


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(_auth: AdminRequired, job_id: str, db: DbSession) -> JobResponse:
    job = db.query(Job).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result_data = None
    if job.result:
        try:
            result_data = json.loads(job.result)
        except json.JSONDecodeError:
            result_data = {"raw": job.result}

    return JobResponse(
        id=job.id,
        type=job.type,
        status=job.status,
        progress=job.progress,
        message=job.message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=result_data,
    )


@router.post("/rebuild-index", response_model=JobResponse, status_code=202)
async def trigger_rebuild_index(
    _auth: AdminRequired,
    db: DbSession,
    temporal: TemporalClientDep,
    request: RebuildIndexRequest | None = None,
) -> JobResponse:
    request = request or RebuildIndexRequest()
    job_id = f"rebuild-{uuid.uuid4().hex[:12]}"

    job = Job(id=job_id, type=JobType.REBUILD_INDEX)
    db.add(job)
    db.commit()

    await temporal.start_workflow(
        RebuildIndexWorkflow.run,
        RebuildIndexWorkflowInput(
            job_id=job_id,
            force=request.force,
            model_name=request.model_name,
        ),
        id=f"rebuild-workflow-{job_id}",
        task_queue=settings.temporal_task_queue_cpu,
    )

    return JobResponse(
        id=job_id,
        type=JobType.REBUILD_INDEX,
        status=JobStatus.PENDING,
        progress=0.0,
        message="Index rebuild queued",
        created_at=job.created_at,
    )


@router.delete("/{job_id}", status_code=204)
async def cancel_job(
    _auth: AdminRequired, job_id: str, db: DbSession, temporal: TemporalClientDep
) -> None:
    job = db.query(Job).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(status_code=400, detail="Cannot cancel completed job")

    if job.workflow_id:
        handle = temporal.get_workflow_handle(job.workflow_id)
        await handle.cancel()

    job.status = JobStatus.CANCELLED
    db.commit()


@router.get("", response_model=JobListResponse)
async def list_jobs(
    _auth: AdminRequired,
    db: DbSession,
    status: JobStatus | None = Query(default=None),
    job_type: JobType | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> JobListResponse:
    query = db.query(Job)

    if status:
        query = query.filter(Job.status == status)
    if job_type:
        query = query.filter(Job.type == job_type)

    total = query.count()
    jobs = query.order_by(Job.created_at.desc()).limit(limit).all()

    return JobListResponse(
        jobs=[
            JobResponse(
                id=j.id,
                type=j.type,
                status=j.status,
                progress=j.progress,
                message=j.message,
                created_at=j.created_at,
                started_at=j.started_at,
                completed_at=j.completed_at,
                result=json.loads(j.result) if j.result else None,
            )
            for j in jobs
        ],
        total=total,
    )


@router.get("/indexes/versions")
async def list_index_versions(
    _auth: AdminRequired,
    db: DbSession,
    limit: int = Query(default=10, ge=1, le=50),
) -> dict:
    builds = db.query(IndexBuild).order_by(IndexBuild.created_at.desc()).limit(limit).all()

    return {
        "versions": [
            {
                "version": b.version,
                "embed_model": b.embed_model,
                "index_type": b.index_type,
                "num_vectors": b.num_vectors,
                "dimension": b.dimension,
                "is_active": b.is_active,
                "created_at": b.created_at.isoformat() if b.created_at else None,
            }
            for b in builds
        ]
    }
