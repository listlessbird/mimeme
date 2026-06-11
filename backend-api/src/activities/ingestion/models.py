from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from domain.adapters.base import FetchRequest
from shared.models import SourceRunStatus, SourceRunTrigger


class FetchSourceInput(BaseModel):
    request: FetchRequest
    source_run_id: int | None = None


class FetchSourceOutput(BaseModel):
    success: bool
    status_code: int | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


class StartSourceRunInput(BaseModel):
    source_id: int
    trigger: SourceRunTrigger = SourceRunTrigger.MANUAL


class StartSourceRunOutput(BaseModel):
    """The run row id plus a config snapshot, so the workflow can plan the fetch
    (build_requests) without any further DB reads."""

    source_run_id: int
    adapter_key: str
    adapter_config: dict[str, Any]
    max_items_per_run: int | None
    dataset: str | None


class FailSourceRunInput(BaseModel):
    source_run_id: int
    error: str


class DiscoverAndQueueInput(BaseModel):
    source_id: int
    source_run_id: int
    adapter_key: str
    dataset: str | None = None
    raw_responses: list[dict[str, Any]]


class DiscoverAndQueueOutput(BaseModel):
    discovered: int
    queued: int
    ingest_job_id: str | None = None


class FinalizeSourceRunInput(BaseModel):
    source_run_id: int


class FinalizeSourceRunOutput(BaseModel):
    status: SourceRunStatus
    discovered: int
    queued: int
    duplicate: int
    failed: int
