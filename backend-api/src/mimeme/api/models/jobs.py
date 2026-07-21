from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from mimeme.db.schema import JobStatus, JobType
from mimeme.domain.job_rules import JobResultPayload


class JobResponse(BaseModel):
    id: str = Field(description="Unique job ID")
    type: JobType = Field(description="Job type")
    status: JobStatus = Field(description="Current status")
    progress: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage (0-100)"
    )
    message: str | None = Field(default=None, description="Status message or error")

    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    result: JobResultPayload | None = Field(default=None, description="Job result data")

    model_config = {"from_attributes": True}


class RebuildIndexRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="Force full rebuild even if incremental is possible",
    )
    model_name: str | None = Field(
        default=None,
        description="Override embedding model (triggers full rebuild)",
    )


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int


class IndexFreshnessResponse(BaseModel):
    desired_generation: int
    active_generation: int
    is_stale: bool
    active_version: str | None
    rebuild_job_id: str | None
    rebuild_target_generation: int | None
    rebuild_claimed_at: datetime | None
    last_dirty_at: datetime | None
    last_dirty_reason: str | None
    last_reconciled_at: datetime | None
