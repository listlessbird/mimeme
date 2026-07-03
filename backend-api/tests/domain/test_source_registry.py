"""Executable spec for the Source registry service (issue 02, slice C).

`SourceRegistry` is the CRUD service over `ingestion_sources` (+ its runs and
items). It owns five operations — create, list, detail, patch, soft-delete —
and two derived-stats concerns:

- **Source-level stats** on list/detail (run count, items discovered,
  duplicates) computed by aggregate query, never stored.
- **Per-run counts** on the detail view's recent runs, derived on read via
  `domain.source_run_accounting.derive_run_accounting` (slice B). The run's
  *status* comes from the stored `source_runs.status` column; only the four
  counts are derived.

Name uniqueness is scoped to live rows (ADR-0003): every lookup-by-name and the
listing carry `WHERE deleted_at IS NULL`, and a soft-deleted name is reusable.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy.orm import Session
from tests.factories import (
    create_image,
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)

from domain.source_registry import (
    DuplicateSourceNameError,
    SourceNotFoundError,
    SourceRegistry,
    UnknownAdapterKeyError,
)
from shared.models import (
    DuplicateReason,
    IngestionSource,
    ProcessingStatus,
    SourceRun,
    SourceRunStatus,
    SourceRunTrigger,
)

UTC = datetime.UTC


def _at(day: int) -> datetime.datetime:
    return datetime.datetime(2026, 1, day, tzinfo=UTC)


# --- create ---------------------------------------------------------------


def test_create_persists_source_and_returns_view(db_session: Session) -> None:
    view = SourceRegistry(db_session).create(
        name="r/memes daily",
        adapter_key="meme_api",
        adapter_config={"subreddits": ["memes", "dankmemes"]},
        dataset="memes",
        schedule_cron="0 * * * *",
        schedule_timezone="UTC",
        max_items_per_run=50,
    )

    row = db_session.query(IngestionSource).filter_by(id=view.id).one()
    assert row.name == "r/memes daily"
    assert row.adapter_key == "meme_api"
    assert row.adapter_config == {"subreddits": ["memes", "dankmemes"]}
    assert view.enabled is True
    assert view.dataset == "memes"
    assert view.created_at is not None


def test_create_rejects_unknown_adapter_key(db_session: Session) -> None:
    with pytest.raises(UnknownAdapterKeyError):
        SourceRegistry(db_session).create(
            name="bad-source",
            adapter_key="not_a_real_adapter",
            adapter_config={},
            dataset=None,
            schedule_cron=None,
            schedule_timezone="UTC",
            max_items_per_run=None,
        )

    # The rejected Source is never written.
    assert db_session.query(IngestionSource).filter_by(name="bad-source").count() == 0


def test_create_rejects_duplicate_live_name(db_session: Session) -> None:
    registry = SourceRegistry(db_session)
    registry.create(
        name="dupe",
        adapter_key="meme_api",
        adapter_config={},
        dataset=None,
        schedule_cron=None,
        schedule_timezone="UTC",
        max_items_per_run=None,
    )

    with pytest.raises(DuplicateSourceNameError):
        registry.create(
            name="dupe",
            adapter_key="meme_api",
            adapter_config={},
            dataset=None,
            schedule_cron=None,
            schedule_timezone="UTC",
            max_items_per_run=None,
        )


def test_create_reuses_name_of_a_soft_deleted_source(db_session: Session) -> None:
    # ADR-0003: soft-deleting frees the name, so recreate-with-same-name works.
    registry = SourceRegistry(db_session)
    first = registry.create(
        name="reuse-me",
        adapter_key="meme_api",
        adapter_config={},
        dataset=None,
        schedule_cron=None,
        schedule_timezone="UTC",
        max_items_per_run=None,
    )
    registry.soft_delete(first.id)

    second = registry.create(
        name="reuse-me",
        adapter_key="meme_api",
        adapter_config={},
        dataset=None,
        schedule_cron=None,
        schedule_timezone="UTC",
        max_items_per_run=None,
    )

    assert second.id != first.id


# --- list -----------------------------------------------------------------


def test_list_excludes_soft_deleted_sources(db_session: Session) -> None:
    registry = SourceRegistry(db_session)
    live = create_ingestion_source(session=db_session, name="live-one")
    deleted = create_ingestion_source(session=db_session, name="gone")
    registry.soft_delete(deleted.id)

    names = {item.name for item in registry.list_sources()}

    assert "live-one" in names
    assert "gone" not in names
    assert live.id is not None


def test_list_derives_stats_by_query(db_session: Session) -> None:
    src = create_ingestion_source(session=db_session, name="stats-src")
    create_source_run(session=db_session, source=src)
    create_source_run(session=db_session, source=src)
    for _ in range(3):
        create_source_item(session=db_session, source=src)
    # Two deduped urls + one genuinely-new url attributed to this source.
    job = create_job(session=db_session)
    image = create_image(session=db_session)
    create_ingest_url(
        session=db_session, job=job, source_id=src.id, duplicate_reason=DuplicateReason.SHA256
    )
    create_ingest_url(
        session=db_session, job=job, source_id=src.id, duplicate_reason=DuplicateReason.PHASH
    )
    create_ingest_url(
        session=db_session,
        job=job,
        source_id=src.id,
        status=ProcessingStatus.DONE,
        image_id=image.id,
        duplicate_reason=None,
    )
    create_ingest_url(session=db_session, job=job, source_id=src.id, status=ProcessingStatus.FAILED)

    item = next(i for i in SourceRegistry(db_session).list_sources() if i.name == "stats-src")

    assert item.stats.run_count == 2
    assert item.stats.items_discovered == 3
    assert item.stats.duplicate_count == 2
    assert item.stats.images_ingested == 1
    assert item.stats.failed_count == 1


def test_list_stats_are_zero_for_a_fresh_source(db_session: Session) -> None:
    create_ingestion_source(session=db_session, name="fresh")

    item = next(i for i in SourceRegistry(db_session).list_sources() if i.name == "fresh")

    assert (
        item.stats.run_count,
        item.stats.items_discovered,
        item.stats.duplicate_count,
        item.stats.images_ingested,
        item.stats.failed_count,
    ) == (0, 0, 0, 0, 0)


# --- detail ---------------------------------------------------------------


def test_get_returns_recent_runs_newest_first(db_session: Session) -> None:
    src = create_ingestion_source(session=db_session)
    create_source_run(session=db_session, source=src, created_at=_at(1))
    newest = create_source_run(session=db_session, source=src, created_at=_at(3))
    create_source_run(session=db_session, source=src, created_at=_at(2))

    detail = SourceRegistry(db_session).get_source(src.id)

    created = [r.created_at for r in detail.recent_runs]
    assert created == sorted(created, reverse=True)
    assert detail.recent_runs[0].id == newest.id


def test_get_recent_run_carries_derived_counts_and_stored_status(db_session: Session) -> None:
    src = create_ingestion_source(session=db_session)
    # Stored status is RUNNING; derive_run_accounting over the urls would say
    # PARTIAL. The view must report the *stored* status and the *derived* counts.
    run = create_source_run(
        session=db_session,
        source=src,
        status=SourceRunStatus.RUNNING,
        trigger_mode=SourceRunTrigger.MANUAL,
    )
    job = create_job(session=db_session)
    create_ingest_url(
        session=db_session,
        job=job,
        source_id=src.id,
        source_run_id=run.id,
        status=ProcessingStatus.DONE,
    )
    create_ingest_url(
        session=db_session,
        job=job,
        source_id=src.id,
        source_run_id=run.id,
        status=ProcessingStatus.DONE,
        duplicate_reason=DuplicateReason.PHASH,
    )
    create_ingest_url(
        session=db_session,
        job=job,
        source_id=src.id,
        source_run_id=run.id,
        status=ProcessingStatus.FAILED,
    )
    for _ in range(4):
        create_source_item(session=db_session, source=src, last_source_run_id=run.id)

    run_view = SourceRegistry(db_session).get_source(src.id).recent_runs[0]

    assert run_view.status == SourceRunStatus.RUNNING
    assert run_view.discovered == 4
    assert run_view.queued == 3
    assert run_view.duplicate == 1
    assert run_view.failed == 1


def test_get_respects_recent_runs_limit(db_session: Session) -> None:
    src = create_ingestion_source(session=db_session)
    for day in range(1, 4):
        create_source_run(session=db_session, source=src, created_at=_at(day))

    detail = SourceRegistry(db_session).get_source(src.id, recent_runs_limit=2)

    assert len(detail.recent_runs) == 2
    assert detail.recent_runs[0].created_at == _at(3)
    assert detail.recent_runs[1].created_at == _at(2)


def test_get_missing_source_raises_not_found(db_session: Session) -> None:
    with pytest.raises(SourceNotFoundError):
        SourceRegistry(db_session).get_source(999_999)


def test_get_soft_deleted_source_raises_not_found(db_session: Session) -> None:
    registry = SourceRegistry(db_session)
    src = create_ingestion_source(session=db_session)
    registry.soft_delete(src.id)

    with pytest.raises(SourceNotFoundError):
        registry.get_source(src.id)


# --- patch ----------------------------------------------------------------


def test_patch_updates_only_provided_fields(db_session: Session) -> None:
    registry = SourceRegistry(db_session)
    src = create_ingestion_source(
        session=db_session,
        schedule_cron="0 * * * *",
        dataset="memes",
        max_items_per_run=50,
    )

    updated = registry.patch(src.id, schedule_cron="0 0 * * *", enabled=False)

    assert updated.schedule_cron == "0 0 * * *"
    assert updated.enabled is False
    # Untouched fields keep their values.
    assert updated.dataset == "memes"
    assert updated.max_items_per_run == 50


def test_patch_can_disable_then_reenable(db_session: Session) -> None:
    registry = SourceRegistry(db_session)
    src = create_ingestion_source(session=db_session, enabled=True)

    assert registry.patch(src.id, enabled=False).enabled is False
    assert registry.patch(src.id, enabled=True).enabled is True


def test_patch_missing_source_raises_not_found(db_session: Session) -> None:
    with pytest.raises(SourceNotFoundError):
        SourceRegistry(db_session).patch(999_999, enabled=False)


def test_patch_soft_deleted_source_raises_not_found(db_session: Session) -> None:
    registry = SourceRegistry(db_session)
    src = create_ingestion_source(session=db_session)
    registry.soft_delete(src.id)

    with pytest.raises(SourceNotFoundError):
        registry.patch(src.id, enabled=False)


# --- soft delete ----------------------------------------------------------


def test_soft_delete_sets_deleted_at_and_hides_from_listing(db_session: Session) -> None:
    registry = SourceRegistry(db_session)
    src = create_ingestion_source(session=db_session, name="to-delete")

    registry.soft_delete(src.id)

    row = db_session.query(IngestionSource).filter_by(id=src.id).one()
    assert row.deleted_at is not None
    assert all(item.name != "to-delete" for item in registry.list_sources())


def test_soft_delete_preserves_runs_and_items(db_session: Session) -> None:
    registry = SourceRegistry(db_session)
    src = create_ingestion_source(session=db_session)
    run = create_source_run(session=db_session, source=src)
    item = create_source_item(session=db_session, source=src)

    registry.soft_delete(src.id)

    assert db_session.query(SourceRun).filter_by(id=run.id).count() == 1
    assert db_session.get(IngestionSource, src.id) is not None
    assert item.id is not None


def test_soft_delete_already_deleted_raises_not_found(db_session: Session) -> None:
    registry = SourceRegistry(db_session)
    src = create_ingestion_source(session=db_session)
    registry.soft_delete(src.id)

    with pytest.raises(SourceNotFoundError):
        registry.soft_delete(src.id)
