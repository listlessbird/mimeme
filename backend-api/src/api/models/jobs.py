from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from shared.models import JobStatus, JobType


class JobResponse(BaseModel):
    id: str = Field(..., description="Unique job ID")
    type: JobType = Field(..., description="Job type")
    status: JobStatus = Field(..., description="Current status")
    progress: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage (0-100)"
    )
    message: str | None = Field(default=None, description="Status message or error")

    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    result: dict[str, Any] | None = Field(default=None, description="Job result data")

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
