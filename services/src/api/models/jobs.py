from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobResponse(BaseModel):
    """Job status and details."""

    id: str = Field(..., description="Unique job ID")
    type: str = Field(..., description="Job type (ingest, annotate, embed, rebuild_index)")
    status: JobStatus = Field(..., description="Current status")
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Progress percentage (0-100)",
    )
    message: str | None = Field(default=None, description="Status message or error")

    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    result: dict[str, Any] | None = Field(
        default=None,
        description="Job result data (structure depends on job type)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "job-abc123",
                "type": "ingest",
                "status": "running",
                "progress": 45.5,
                "message": "Processing image 46 of 100",
                "created_at": "2025-01-15T10:30:00Z",
                "started_at": "2025-01-15T10:30:05Z",
                "completed_at": None,
                "result": None,
            }
        }
    }


class RebuildIndexRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="Force full rebuild even if incremental is possible",
    )
    model_name: str | None = Field(
        default=None,
        description="Override embedding model (triggers full rebuild)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "force": False,
            }
        }
    }


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int
