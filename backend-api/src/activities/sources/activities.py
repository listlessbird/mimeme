from __future__ import annotations

import datetime
import time

import structlog
from iso8601 import UTC
from temporalio import activity

from activities.sources import (
    CompleteSourceRunInput,
    CreateSourceIngestInput,
    FilterSeenItemsInput,
    FilterSeenItemsOutput,
    PersistSourceItemsInput,
    PersistSourceItemsOutput,
)
from activities.sources.models import (
    CreateSourceRunInput,
    CreateSourceRunOutput,
    FetchSourceItemsInput,
    FetchSourceItemsOutput,
)
from activities.sources.registry import get_adapter
from shared.config import settings
from shared.db import session_scope
from shared.logging import emit_activity_event
from shared.models import IngestURL, Job, JobType
from shared.models.orm import IngestionSource, SourceItem, SourceRun, SourceRunStatus

log = structlog.get_logger()

_SECRET_REF_PREFIX = "secret-ref:"


@activity.defn
async def create_source_run_activity(input: CreateSourceRunInput) -> CreateSourceRunOutput:
    started = time.monotonic()
    try:
        with session_scope() as session:
            run = SourceRun(
                source_id=input.source_id,
                trigger_mode=input.trigger_mode,
                status=SourceRunStatus.RUNNING,
                started_at=datetime.datetime.now(UTC),
            )
            session.add(run)
            session.flush()
            run_id = run.id

        emit_activity_event(
            log=log,
            activity_name="create_source_run_activity",
            started_at=started,
            outcome="success",
            source_id=input.source_id,
            source_run_id=run_id,
        )
        return CreateSourceRunOutput(source_run_id=run_id)
    except Exception as ex:
        emit_activity_event(
            log=log,
            activity_name="create_source_run_activity",
            started_at=started,
            outcome="error",
            source_id=input.source_id,
            error=str(ex),
        )
        raise


@activity.defn
async def fetch_source_items_activity(input: FetchSourceItemsInput) -> FetchSourceItemsOutput:
    started = time.monotonic()
    try:
        adapter = get_adapter(input.adapter_key)
        expected_secret_refs = {
            f"{_SECRET_REF_PREFIX}{secret_name}" for secret_name in adapter.secret_ref_names
        }
        provided_secret_refs = set(input.secret_refs or [])

        missing_secret_refs = expected_secret_refs - provided_secret_refs

        if missing_secret_refs:
            raise ValueError(
                f"Missing secret refs for {input.adapter_key!r}: {sorted(missing_secret_refs)}"
            )

        secrets: dict[str, str] = {}
        for setting_name in adapter.secret_ref_names:
            secret_value = getattr(settings, setting_name, "")
            if not secret_value:
                raise ValueError(
                    f"Secret ref {_SECRET_REF_PREFIX}{setting_name!s} "
                    f"for {input.adapter_key!r} is not configured"
                )
            secrets[setting_name] = secret_value

        output = adapter.fetch_latest(
            adapter_cfg=input.adapter_config,
            max_items=input.max_items_per_run,
            secrets=secrets,
        )

        emit_activity_event(
            log=log,
            activity_name="fetch_source_items_activity",
            started_at=started,
            outcome="success",
            source_id=input.source_id,
            adapter_key=input.adapter_key,
            item_count=len(output.items),
            skipped=output.skipped,
        )
        return output
    except Exception as ex:
        emit_activity_event(
            log=log,
            activity_name="fetch_source_items_activity",
            started_at=started,
            outcome="error",
            source_id=input.source_id,
            adapter_key=input.adapter_key,
            error=str(ex),
        )
        raise


@activity.defn
async def filter_seen_items_activity(input: FilterSeenItemsInput) -> FilterSeenItemsOutput:
    started = time.monotonic()
    try:
        external_ids = [item.external_item_id for item in input.items]

        with session_scope() as session:
            existing = set(
                row[0]
                for row in session.query(SourceItem.external_item_id)
                .filter(
                    SourceItem.source_id == input.source_id,
                    SourceItem.external_item_id.in_(external_ids),
                )
                .all()
            )

        new_items = [item for item in input.items if item.external_item_id not in existing]
        seen_count = len(input.items) - len(new_items)

        emit_activity_event(
            log=log,
            activity_name="filter_seen_items_activity",
            started_at=started,
            outcome="success",
            source_id=input.source_id,
            total=len(input.items),
            new=len(new_items),
            seen=seen_count,
        )
        return FilterSeenItemsOutput(new_items=new_items, seen_count=seen_count)
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="filter_seen_items_activity",
            started_at=started,
            outcome="error",
            source_id=input.source_id,
            error=str(exc),
        )
        raise


@activity.defn
async def persist_source_items_activity(input: PersistSourceItemsInput) -> PersistSourceItemsOutput:
    started = time.monotonic()
    try:
        source_item_ids: list[int] = []
        now = datetime.datetime.now(UTC)

        with session_scope() as session:
            for item_data in input.items:
                existing = (
                    session.query(SourceItem)
                    .filter_by(
                        source_id=input.source_id,
                        external_item_id=item_data.external_item_id,
                    )
                    .first()
                )

                if existing:
                    existing.last_source_run_id = input.source_run_id
                    existing.last_seen_at = now
                    source_item_ids.append(existing.id)
                else:
                    si = SourceItem(
                        source_id=input.source_id,
                        last_source_run_id=input.source_run_id,
                        external_item_id=item_data.external_item_id,
                        canonical_item_url=item_data.canonical_item_url,
                        canonical_media_id=item_data.canonical_media_id,
                        title=item_data.title,
                        published_at=item_data.published_at,
                        raw_metadata=item_data.raw_metadata,
                    )
                    session.add(si)
                    session.flush()
                    source_item_ids.append(si.id)

        emit_activity_event(
            log=log,
            activity_name="persist_source_items_activity",
            started_at=started,
            outcome="success",
            source_id=input.source_id,
            persisted=len(source_item_ids),
        )
        return PersistSourceItemsOutput(source_item_ids=source_item_ids)
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="persist_source_items_activity",
            started_at=started,
            outcome="error",
            source_id=input.source_id,
            error=str(exc),
        )
        raise


@activity.defn
async def create_source_ingest_activity(input: CreateSourceIngestInput) -> str:
    started = time.monotonic()
    try:
        job_id = input.job_id

        with session_scope() as session:
            job = Job(id=job_id, type=JobType.INGEST)
            session.add(job)
            session.flush()

            for item_data, source_item_id in zip(input.items, input.source_item_ids):
                ingest_url = IngestURL(
                    job_id=job_id,
                    url=item_data.fetch_url,
                    source_id=input.source_id,
                    source_run_id=input.source_run_id,
                    source_item_id=source_item_id,
                )
                session.add(ingest_url)

        emit_activity_event(
            log=log,
            activity_name="create_source_ingest_activity",
            started_at=started,
            outcome="success",
            source_id=input.source_id,
            job_id=job_id,
            urls_created=len(input.items),
        )
        return job_id
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="create_source_ingest_activity",
            started_at=started,
            outcome="error",
            source_id=input.source_id,
            error=str(exc),
        )
        raise


@activity.defn
async def complete_source_run_activity(input: CompleteSourceRunInput) -> None:
    started = time.monotonic()
    try:
        now = datetime.datetime.now(UTC)

        with session_scope() as session:
            run = session.query(SourceRun).filter_by(id=input.source_run_id).first()
            if not run:
                raise ValueError(f"SourceRun {input.source_run_id} not found")

            run.status = input.status
            run.error_message = input.error_message
            run.summary = input.summary
            run.completed_at = now

            if input.status == SourceRunStatus.COMPLETED.value:
                source = session.query(IngestionSource).filter_by(id=input.source_id).first()
                if source:
                    source.last_successful_run_at = now

        emit_activity_event(
            log=log,
            activity_name="complete_source_run_activity",
            started_at=started,
            outcome="success",
            source_run_id=input.source_run_id,
            source_id=input.source_id,
            status=input.status,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="complete_source_run_activity",
            started_at=started,
            outcome="error",
            source_run_id=input.source_run_id,
            error=str(exc),
        )
        raise


@activity.defn
async def load_source_config_activity(source_id: int) -> dict:
    started = time.monotonic()
    try:
        with session_scope() as session:
            source = session.query(IngestionSource).filter_by(id=source_id).first()
            if not source:
                raise ValueError(f"IngestionSource {source_id} not found")
            if source.deleted_at is not None:
                raise ValueError(f"IngestionSource {source_id} has been deleted")

            config = {
                "source_id": source.id,
                "name": source.name,
                "adapter_key": source.adapter_key,
                "adapter_config": source.adapter_config or {},
                "max_items_per_run": source.max_items_per_run,
                "dataset": source.dataset,
                "default_tags": source.default_tags,
                "enabled": source.enabled,
            }

        emit_activity_event(
            log=log,
            activity_name="load_source_config_activity",
            started_at=started,
            outcome="success",
            source_id=source_id,
        )
        return config
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="load_source_config_activity",
            started_at=started,
            outcome="error",
            source_id=source_id,
            error=str(exc),
        )
        raise
