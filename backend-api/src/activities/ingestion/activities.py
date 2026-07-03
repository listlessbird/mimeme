from __future__ import annotations

import time
from datetime import UTC, datetime

import httpx
import structlog
from sqlalchemy import func, update
from sqlalchemy.orm import Session
from temporalio import activity

from activities.ingestion.models import (
    DiscoverAndQueueInput,
    DiscoverAndQueueOutput,
    FailSourceRunInput,
    FetchSourceInput,
    FetchSourceOutput,
    FinalizeSourceRunInput,
    FinalizeSourceRunOutput,
    StartSourceRunInput,
    StartSourceRunOutput,
)
from domain.adapters.base import DiscoveredItem
from domain.adapters.registry import get_adapter
from domain.job_lifecycle import JobLifecycle, SourceIngestItem
from domain.source_item_dedup import dedup_source_items
from domain.source_run_accounting import UrlOutcome, derive_run_accounting
from shared.db import session_scope
from shared.logging import emit_activity_event
from shared.models import SourceRunStatus
from shared.models.orm import IngestionSource, IngestURL, SourceItem, SourceRun

log = structlog.get_logger()

_RETRYABLE_4XX = frozenset({408, 425, 429})

_DEFAULT_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ),
    "accept": "application/json",
}


@activity.defn
def fetch_source_activity(input: FetchSourceInput) -> FetchSourceOutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None
    status_code: int | None = None
    request = input.request

    log.info(
        "activity_step",
        activity_name="fetch_source_activity",
        step="start_fetch",
        source_run_id=input.source_run_id,
        method=str(request.method),
        url=request.url,
    )

    try:
        headers = {**_DEFAULT_HEADERS, **request.headers}
        with httpx.Client(
            timeout=request.timeout_seconds,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = client.request(str(request.method), request.url)
            status_code = response.status_code
            response.raise_for_status()

            try:
                raw = response.json()
            except ValueError as e:
                outcome = "failed"
                error_message = f"invalid_json:{e}"
                return FetchSourceOutput(
                    success=False, status_code=status_code, error=error_message
                )

            if not isinstance(raw, dict):
                outcome = "failed"
                error_message = f"unexpected_json_shape:{type(raw).__name__}"
                return FetchSourceOutput(
                    success=False, status_code=status_code, error=error_message
                )

            return FetchSourceOutput(success=True, status_code=status_code, raw=raw)

    except httpx.HTTPStatusError as e:
        if e.response is not None:
            status_code = e.response.status_code

        if (
            status_code is not None
            and 400 <= status_code < 500
            and status_code not in _RETRYABLE_4XX
        ):
            log.info(
                "activity_step",
                activity_name="fetch_source_activity",
                step="terminal_http_error",
                source_run_id=input.source_run_id,
                status_code=status_code,
                error=str(e),
            )
            outcome = "failed"
            error_message = str(e)
            return FetchSourceOutput(success=False, status_code=status_code, error=str(e))

        log.info(
            "activity_step",
            activity_name="fetch_source_activity",
            step="retryable_http_error",
            source_run_id=input.source_run_id,
            status_code=status_code,
            error=str(e),
        )
        outcome = "error"
        error_message = str(e)
        raise
    except httpx.TransportError as e:
        log.info(
            "activity_step",
            activity_name="fetch_source_activity",
            step="retryable_transport_error",
            source_run_id=input.source_run_id,
            error=str(e),
        )
        outcome = "error"
        error_message = str(e)
        raise
    except Exception as e:
        outcome = "error"
        error_message = str(e)
        raise
    finally:
        emit_activity_event(
            log=log,
            activity_name="fetch_source_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            source_run_id=input.source_run_id,
            url=request.url,
            status_code=status_code,
        )


def _run_counts(session: Session, source_run_id: int) -> tuple[int, int]:
    discovered = (
        session.query(func.count(SourceItem.id))
        .filter(SourceItem.last_source_run_id == source_run_id)
        .scalar()
        or 0
    )
    queued = (
        session.query(func.count(IngestURL.id))
        .filter(IngestURL.source_run_id == source_run_id)
        .scalar()
        or 0
    )
    return discovered, queued


@activity.defn
def discover_and_queue_activity(input: DiscoverAndQueueInput) -> DiscoverAndQueueOutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None
    queued = 0

    try:
        with session_scope() as session:
            run = session.get(SourceRun, input.source_run_id)
            if run is None:
                raise ValueError(f"source_run {input.source_run_id} not found")

            if run.ingest_job_id is not None:
                discovered, queued = _run_counts(session, input.source_run_id)
                return DiscoverAndQueueOutput(
                    discovered=discovered, queued=queued, ingest_job_id=run.ingest_job_id
                )

            adapter = get_adapter(input.adapter_key)
            discovered_items: list[DiscoveredItem] = []
            for raw in input.raw_responses:
                discovered_items.extend(adapter.parse(raw))

            seen_ids = {
                external_item_id
                for (external_item_id,) in (
                    session.query(SourceItem.external_item_id)
                    .filter(SourceItem.source_id == input.source_id)
                    .all()
                )
            }

            dedup = dedup_source_items(discovered_items, seen_ids=seen_ids)
            now = datetime.now(UTC)

            seen_external_ids = [item.external_item_id for item in dedup.already_seen]

            if seen_external_ids:
                session.execute(
                    update(SourceItem)
                    .where(
                        SourceItem.source_id == input.source_id,
                        SourceItem.external_item_id.in_(seen_external_ids),
                    )
                    .values(last_seen_at=now, last_source_run_id=input.source_run_id)
                )

            new_pairs: list[tuple[DiscoveredItem, SourceItem]] = []
            for item in dedup.new:
                source_item = SourceItem(
                    source_id=input.source_id,
                    last_source_run_id=input.source_run_id,
                    external_item_id=item.external_item_id,
                    canonical_item_url=item.canonical_item_url,
                    canonical_image_url=item.canonical_image_url,
                    title=item.title,
                    raw_metadata=item.raw_metadata,
                    first_seen_at=now,
                    last_seen_at=now,
                )
                session.add(source_item)
                new_pairs.append((item, source_item))

            session.flush()  # assign source_item ids for provenance

            ingest_job_id: str | None = None
            if new_pairs:
                creation = JobLifecycle(session).create_source_ingest_job(
                    dataset=input.dataset,
                    items=[
                        SourceIngestItem(
                            url=item.media_url,
                            source_id=input.source_id,
                            source_run_id=input.source_run_id,
                            source_item_id=source_item.id,
                        )
                        for item, source_item in new_pairs
                    ],
                )
                ingest_job_id = creation.job_id
                run.ingest_job_id = ingest_job_id

            queued = len(dedup.new)
            return DiscoverAndQueueOutput(
                discovered=len(dedup.new) + len(dedup.already_seen),
                queued=queued,
                ingest_job_id=ingest_job_id,
            )
    except Exception as e:
        outcome = "error"
        error_message = str(e)
        raise
    finally:
        emit_activity_event(
            log=log,
            activity_name="discover_and_queue_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            source_id=input.source_id,
            source_run_id=input.source_run_id,
            queued=queued,
        )


@activity.defn
def finalize_source_run_activity(input: FinalizeSourceRunInput) -> FinalizeSourceRunOutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None

    try:
        with session_scope() as session:
            run = session.get(SourceRun, input.source_run_id)
            if run is None:
                raise ValueError(f"source_run {input.source_run_id} not found")

            discovered = (
                session.query(func.count(SourceItem.id))
                .filter(SourceItem.last_source_run_id == input.source_run_id)
                .scalar()
                or 0
            )
            outcomes = [
                UrlOutcome(status=status, duplicate_reason=duplicate_reason)
                for status, duplicate_reason in (
                    session.query(IngestURL.status, IngestURL.duplicate_reason)
                    .filter(IngestURL.source_run_id == input.source_run_id)
                    .all()
                )
            ]

            accounting = derive_run_accounting(discovered_items=discovered, url_outcomes=outcomes)

            run.status = accounting.status
            if run.completed_at is None:
                run.completed_at = datetime.now(UTC)

            return FinalizeSourceRunOutput(
                status=accounting.status,
                discovered=accounting.discovered,
                queued=accounting.queued,
                duplicate=accounting.duplicate,
                failed=accounting.failed,
            )
    except Exception as e:
        outcome = "error"
        error_message = str(e)
        raise
    finally:
        emit_activity_event(
            log=log,
            activity_name="finalize_source_run_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            source_run_id=input.source_run_id,
        )


@activity.defn
def start_source_run_activity(input: StartSourceRunInput) -> StartSourceRunOutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None

    try:
        with session_scope() as session:
            source = (
                session.query(IngestionSource)
                .filter(
                    IngestionSource.id == input.source_id,
                    IngestionSource.deleted_at.is_(None),
                )
                .one_or_none()
            )
            if source is None:
                raise ValueError(f"source {input.source_id} not found")

            run = SourceRun(
                source_id=source.id,
                trigger_mode=input.trigger,
                status=SourceRunStatus.RUNNING,
                started_at=datetime.now(UTC),
            )
            session.add(run)
            session.flush()

            return StartSourceRunOutput(
                source_run_id=run.id,
                adapter_key=source.adapter_key,
                adapter_config=source.adapter_config,
                max_items_per_run=source.max_items_per_run,
                dataset=source.dataset,
            )
    except Exception as e:
        outcome = "error"
        error_message = str(e)
        raise
    finally:
        emit_activity_event(
            log=log,
            activity_name="start_source_run_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            source_id=input.source_id,
        )


@activity.defn
def fail_source_run_activity(input: FailSourceRunInput) -> None:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None

    try:
        with session_scope() as session:
            run = session.get(SourceRun, input.source_run_id)
            if run is None:
                return

            run.status = SourceRunStatus.FAILED
            run.error_message = input.error[:1000] if input.error else input.error
            if run.completed_at is None:
                run.completed_at = datetime.now(UTC)
    except Exception as e:
        outcome = "error"
        error_message = str(e)
        raise
    finally:
        emit_activity_event(
            log=log,
            activity_name="fail_source_run_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            source_run_id=input.source_run_id,
        )
