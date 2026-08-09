from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, NotRequired, TypedDict

from pydantic import BaseModel, Field

from mimeme.db.schema import DuplicateReason, ProcessingStatus, SourceRunStatus, SourceRunTrigger
from mimeme.ingest import Source
from mimeme.source import SourceItemIngestState


class MemeApiAdapterConfig(TypedDict):
    subreddits: Annotated[list[str], Field(min_length=1)]
    min_score: NotRequired[Annotated[int, Field(ge=0)]]
    max_items_per_run: NotRequired[Annotated[int | None, Field(ge=0)]]


class TumblrAdapterConfig(TypedDict):
    tags: NotRequired[Annotated[list[str], Field(min_length=1)]]
    min_note_count: NotRequired[Annotated[int, Field(ge=0)]]
    max_items_per_run: NotRequired[Annotated[int | None, Field(ge=0)]]


AdapterConfig = MemeApiAdapterConfig | TumblrAdapterConfig


class MemeApiRawMetadata(TypedDict, total=False):
    author: str | None
    title: str | None
    ups: int | None
    subreddit: str | None
    preview: str | list[str] | None
    postLink: str | None


class CreateSourceRequest(BaseModel):
    name: str = Field(description="Unique (among live Sources) display name")
    adapter_key: str = Field(description="Adapter the system supports, e.g. 'meme_api' or 'tumblr_tagged'")
    adapter_config: AdapterConfig
    dataset: str | None = None
    schedule_cron: str | None = None
    schedule_timezone: str = "UTC"
    max_items_per_run: int | None = None
    enabled: bool = True


class UpdateSourceRequest(BaseModel):
    adapter_config: AdapterConfig | None = None
    dataset: str | None = None
    schedule_cron: str | None = None
    schedule_timezone: str = "UTC"
    max_items_per_run: int | None = None
    enabled: bool = True


class SourceResponse(BaseModel):
    id: int
    name: str
    adapter_key: str
    adapter_config: AdapterConfig
    dataset: str | None
    schedule_cron: str | None
    schedule_timezone: str
    max_items_per_run: int | None
    enabled: bool
    created_at: datetime
    updated_at: datetime


class SourceStatsResponse(BaseModel):
    run_count: int
    items_discovered: int
    duplicate_count: int
    images_ingested: int
    failed_count: int


class SourceRunResponse(BaseModel):
    id: int
    trigger_mode: SourceRunTrigger
    status: SourceRunStatus
    ingest_job_id: str | None
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    discovered: int
    queued: int
    duplicate: int
    failed: int


class SourceListItemResponse(SourceResponse):
    stats: SourceStatsResponse


class SourceDetailResponse(SourceResponse):
    stats: SourceStatsResponse
    recent_runs: list[SourceRunResponse]


class SourceListResponse(BaseModel):
    sources: list[SourceListItemResponse]
    total: int


class TriggerRunResponse(BaseModel):
    workflow_id: str
    message: str


class RetryResponse(BaseModel):
    job_id: str
    workflow_id: str
    queued: int
    message: str


class SourceItemResponse(BaseModel):
    id: int
    external_item_id: str
    title: str | None
    raw_metadata: dict[str, Any] | None
    thumbnail_url: str | None
    first_seen_at: datetime
    last_seen_at: datetime
    ingest_state: SourceItemIngestState
    resolved_image_id: int | None
    duplicate_reason: DuplicateReason | None
    duplicate_of_image_id: int | None
    attempt_status: ProcessingStatus | None
    attempt_error_message: str | None
    attempt_source_run_id: int | None
    media_url: str | None


class SourceItemListResponse(BaseModel):
    items: list[SourceItemResponse]
    total: int
    limit: int
    offset: int
    state_counts: dict[str, int]


class RunItemResponse(BaseModel):
    id: int
    input: Source
    source_item_id: int | None
    external_item_id: str | None
    title: str | None
    status: ProcessingStatus
    error_message: str | None
    duplicate_reason: DuplicateReason | None
    image_id: int | None
    thumbnail_url: str | None


class RunItemListResponse(BaseModel):
    items: list[RunItemResponse]
    total: int
    limit: int
    offset: int
