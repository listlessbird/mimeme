from __future__ import annotations

import datetime
from enum import StrEnum
from http import HTTPMethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mimeme.db.schema import (
    DuplicateReason,
    ProcessingStatus,
    SourceRunStatus,
    SourceRunTrigger,
)
from mimeme.ingest.model import ItemRef, Source


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Error(Exception):
    pass


class SourceNotFound(Error):
    pass


class DuplicateSourceName(Error):
    pass


class RunNotFound(Error):
    pass


class SourceItemNotFound(Error):
    pass


class NothingToRetry(Error):
    pass


class Retryable(Error):
    """Transient fetch/infra failure. The activity should retry."""


class FetchRequest(_Frozen):
    url: str
    method: HTTPMethod = HTTPMethod.GET
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 30.0


class RawResponse(_Frozen):
    success: bool
    status_code: int | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


class DiscoveredItem(_Frozen):
    external_item_id: str
    media_url: str
    canonical_item_url: str | None = None
    canonical_image_url: str | None = None
    title: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class SourceItemDedup(_Frozen):
    new: list[DiscoveredItem]
    already_seen: list[DiscoveredItem]


def dedup_source_items(discovered: list[DiscoveredItem], *, seen_ids: set[str]) -> SourceItemDedup:
    new: list[DiscoveredItem] = []
    already_seen: list[DiscoveredItem] = []
    collapsed: set[str] = set()

    for item in discovered:
        external_item_id = item.external_item_id
        if external_item_id in collapsed:
            continue
        collapsed.add(external_item_id)
        if external_item_id in seen_ids:
            already_seen.append(item)
        else:
            new.append(item)

    return SourceItemDedup(new=new, already_seen=already_seen)


class UrlOutcome(_Frozen):
    status: ProcessingStatus
    duplicate_reason: DuplicateReason | None = None


class RunAccounting(_Frozen):
    status: SourceRunStatus
    discovered: int
    queued: int
    duplicate: int
    failed: int


def derive_run_accounting(*, discovered_items: int, url_outcomes: list[UrlOutcome]) -> RunAccounting:
    queued = len(url_outcomes)
    duplicate = sum(1 for outcome in url_outcomes if outcome.duplicate_reason is not None)
    failed = sum(1 for outcome in url_outcomes if outcome.status == ProcessingStatus.FAILED)

    if failed == 0:
        status = SourceRunStatus.COMPLETED
    elif failed == queued:
        status = SourceRunStatus.FAILED
    else:
        status = SourceRunStatus.PARTIAL

    return RunAccounting(
        status=status,
        discovered=discovered_items,
        queued=queued,
        duplicate=duplicate,
        failed=failed,
    )


class SourceView(_Frozen):
    id: int
    name: str
    adapter_key: str
    adapter_config: dict[str, Any]
    dataset: str | None
    schedule_cron: str | None
    schedule_timezone: str
    max_items_per_run: int | None
    enabled: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime


class SourceStats(_Frozen):
    run_count: int
    items_discovered: int
    duplicate_count: int
    images_ingested: int
    failed_count: int


class SourceRunView(_Frozen):
    id: int
    trigger_mode: SourceRunTrigger
    status: SourceRunStatus
    ingest_job_id: str | None
    error_message: str | None
    started_at: datetime.datetime | None
    completed_at: datetime.datetime | None
    created_at: datetime.datetime
    discovered: int
    queued: int
    duplicate: int
    failed: int


class SourceListItem(SourceView):
    stats: SourceStats


class SourceDetail(SourceView):
    stats: SourceStats
    recent_runs: list[SourceRunView]


class SourceItemIngestState(StrEnum):
    INGESTED = "ingested"
    DEDUPED = "deduped"
    FAILED = "failed"
    IN_FLIGHT = "in_flight"
    UNKNOWN = "unknown"


class SourceItemView(_Frozen):
    id: int
    external_item_id: str
    title: str | None
    raw_metadata: dict[str, Any] | None
    thumbnail_url: str | None
    first_seen_at: datetime.datetime
    last_seen_at: datetime.datetime
    ingest_state: SourceItemIngestState
    resolved_image_id: int | None
    duplicate_reason: DuplicateReason | None
    duplicate_of_image_id: int | None
    attempt_status: ProcessingStatus | None
    attempt_error_message: str | None
    attempt_source_run_id: int | None
    media_url: str | None


class SourceItemsPage(_Frozen):
    items: list[SourceItemView]
    total: int
    limit: int
    offset: int
    state_counts: dict[str, int]


class RunItemView(_Frozen):
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


class RunItemsPage(_Frozen):
    items: list[RunItemView]
    total: int
    limit: int
    offset: int


class RetryPlan(_Frozen):
    job_id: str
    workflow_id: str
    source_run_ids: list[int]
    dataset: str | None
    items: list[ItemRef]
    count: int


class DiscoverInput(_Frozen):
    source_id: int
    trigger: SourceRunTrigger = SourceRunTrigger.MANUAL


class DiscoverResult(_Frozen):
    source_run_id: int
    ingest_job_id: str | None
    dataset: str | None
    items: list[ItemRef]
    discovered: int
    queued: int


class FinishInput(_Frozen):
    source_run_id: int
    error: str | None = None


class FinishResult(_Frozen):
    status: SourceRunStatus
    discovered: int
    queued: int
    duplicate: int
    failed: int


class SyncInput(_Frozen):
    source_id: int
    trigger: SourceRunTrigger = SourceRunTrigger.MANUAL


class SyncResult(_Frozen):
    source_run_id: int
    status: SourceRunStatus
    discovered: int
    queued: int
    duplicate: int
    failed: int
    ingest_job_id: str | None = None


class RetryInput(_Frozen):
    job_id: str
    dataset: str | None
    source_run_ids: list[int]
    items: list[ItemRef]


class RetryResult(_Frozen):
    job_id: str
    source_run_ids: list[int]
    statuses: list[SourceRunStatus]
