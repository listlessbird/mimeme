from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from shared.models import SourceRunStatus, SourceRunTrigger


class CreateSourceRequest(BaseModel):
    name: str = Field(description="Unique (among live Sources) display name")
    adapter_key: str = Field(description="Adapter the system supports, e.g. 'meme_api'")
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    dataset: str | None = None
    schedule_cron: str | None = None
    schedule_timezone: str = "UTC"
    max_items_per_run: int | None = None
    enabled: bool = True


class UpdateSourceRequest(BaseModel):
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    dataset: str | None = None
    schedule_cron: str | None = None
    schedule_timezone: str = "UTC"
    max_items_per_run: int | None = None
    enabled: bool = True


class SourceResponse(BaseModel):
    id: int
    name: str
    adapter_key: str
    adapter_config: dict[str, Any]
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


class SourceRunResponse(BaseModel):
    id: int
    trigger_mode: SourceRunTrigger
    status: SourceRunStatus
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
