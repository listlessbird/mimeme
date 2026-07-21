from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

from mimeme.db.schema import JobStatus, JobType

JOB_ERROR_LIMIT = 2000
INGEST_URL_ERROR_LIMIT = 1000


class JobLifecycleNotFoundError(Exception):
    pass


class JobLifecycleInvalidStateError(Exception):
    pass


class IngestJobResultPayload(BaseModel, frozen=True):
    processed: int = Field(ge=0)
    failed: int = Field(ge=0)
    duplicates: int = Field(ge=0)


class RebuildJobResultPayload(BaseModel, frozen=True):
    version: str
    num_vectors: int = Field(ge=0)
    dimension: int = Field(ge=0)
    removed_versions: list[str]
    text_num_vectors: int | None = Field(default=None, ge=0)
    skipped: bool = False
    skip_reason: str | None = None


class RawJobResultPayload(BaseModel, frozen=True):
    raw: str


type JobResultPayload = IngestJobResultPayload | RebuildJobResultPayload | RawJobResultPayload


class JobView(BaseModel, frozen=True):
    id: str
    type: JobType
    status: JobStatus
    progress: float
    message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    result: JobResultPayload | None


class JobList(BaseModel, frozen=True):
    jobs: list[JobView]
    total: int


class IngestJobCreation(BaseModel, frozen=True):
    job_id: str
    workflow_id: str
    queued: int
    duplicates: int
    dataset: str | None
    tags: list[str]
    callback_url: str | None


class RebuildJobCreation(BaseModel, frozen=True):
    job: JobView
    workflow_id: str
    force: bool
    model_name: str
    index_type: str


class JobCancellation(BaseModel, frozen=True):
    workflow_id: str | None


class IndexBuildView(BaseModel, frozen=True, from_attributes=True):
    version: str
    embed_model: str | None
    index_type: str | None
    num_vectors: int | None
    dimension: int | None
    is_active: bool
    created_at: datetime | None


class JobRowData(BaseModel, frozen=True, from_attributes=True):
    id: str
    type: JobType
    status: JobStatus
    progress: float
    message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    result: str | None


def mint_ingest_job() -> tuple[str, str]:
    job_id = f"ingest-{uuid.uuid4().hex[:12]}"
    return job_id, f"ingest-workflow-{job_id}"


def mint_rebuild_job() -> tuple[str, str]:
    job_id = f"rebuild-{uuid.uuid4().hex[:12]}"
    return job_id, f"rebuild-workflow-{job_id}"


def dedup_urls(urls: list[str]) -> tuple[list[str], int]:
    unique_urls = list(dict.fromkeys(urls))
    return unique_urls, len(urls) - len(unique_urls)


def ensure_cancellable(status: JobStatus) -> None:
    if status in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise JobLifecycleInvalidStateError("Cannot cancel completed job")


def derive_completion_status(*, failed: int) -> JobStatus:
    return JobStatus.COMPLETED if failed == 0 else JobStatus.FAILED


def truncate_error(error: str, limit: int) -> str:
    return error[:limit]


def parse_result(job_type: JobType, result: str | None) -> JobResultPayload | None:
    if result is None:
        return None

    try:
        if job_type == JobType.REBUILD_INDEX:
            return RebuildJobResultPayload.model_validate_json(result)
        return IngestJobResultPayload.model_validate_json(result)
    except ValidationError:
        return RawJobResultPayload(raw=result)


def project_job(row: JobRowData) -> JobView:
    return JobView(
        id=row.id,
        type=row.type,
        status=row.status,
        progress=row.progress,
        message=row.message,
        created_at=row.created_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
        result=parse_result(row.type, row.result),
    )
