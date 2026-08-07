from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from mimeme.db.schema import DuplicateReason, IngestStage, ProcessingStatus, SourceRunTrigger
from mimeme.ingest import Source
from mimeme.ingest.browse import Outcome


class IngestionRowResponse(BaseModel):
    ingest_url_id: int
    input: Source
    job_id: str
    source_run_id: int | None
    source_id: int | None
    source_name: str | None
    trigger: SourceRunTrigger
    stage: IngestStage
    status: ProcessingStatus
    outcome: Outcome
    duplicate_reason: DuplicateReason | None
    duplicate_of_image_id: int | None
    resolved_image_id: int | None
    dataset: str | None
    thumbnail_url: str | None
    error_message: str | None
    created_at: datetime
    stage_updated_at: datetime | None


class IngestionListResponse(BaseModel):
    rows: list[IngestionRowResponse]
    total: int
    limit: int
    offset: int


class IngestionDetailResponse(IngestionRowResponse):
    image_id: int | None


class IngestionLogEntryResponse(BaseModel):
    time: str
    level: str | None = None
    event: str | None = None
    activity_name: str | None = None
    step: str | None = None
    outcome: str | None = None
    duration_ms: int | None = None
    error: str | None = None
    attempt: int | None = None
    workflow_id: str | None = None


class IngestionLogsResponse(BaseModel):
    available: bool
    workflow_id: str | None = None
    entries: list[IngestionLogEntryResponse]
