from __future__ import annotations

import uuid

from pydantic import ValidationError

from mimeme.db.schema import JobStatus, JobType
from mimeme.job.model import (
    INGEST_URL_ERROR_LIMIT,
    JOB_ERROR_LIMIT,
    IngestResult,
    InvalidState,
    RawResult,
    RebuildResult,
    Result,
    RowData,
    View,
)

__all__ = [
    "INGEST_URL_ERROR_LIMIT",
    "JOB_ERROR_LIMIT",
    "derive_completion_status",
    "ensure_cancellable",
    "mint_ingest",
    "mint_rebuild",
    "parse_result",
    "project",
    "truncate",
]

_TERMINAL = (JobStatus.COMPLETED, JobStatus.FAILED)


def mint_ingest() -> tuple[str, str]:
    job_id = f"ingest-{uuid.uuid4().hex[:12]}"
    return job_id, f"ingest-workflow-{job_id}"


def mint_rebuild() -> tuple[str, str]:
    job_id = f"rebuild-{uuid.uuid4().hex[:12]}"
    return job_id, f"rebuild-workflow-{job_id}"


def ensure_cancellable(status: JobStatus) -> None:
    if status in _TERMINAL:
        raise InvalidState("Cannot cancel completed job")


def derive_completion_status(*, failed: int) -> JobStatus:
    return JobStatus.COMPLETED if failed == 0 else JobStatus.FAILED


def truncate(error: str, limit: int) -> str:
    return error[:limit]


def parse_result(job_type: JobType, result: str | None) -> Result | None:
    if result is None:
        return None
    try:
        if job_type == JobType.REBUILD_INDEX:
            return RebuildResult.model_validate_json(result)
        return IngestResult.model_validate_json(result)
    except ValidationError:
        return RawResult(raw=result)


def project(row: RowData) -> View:
    return View(
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
