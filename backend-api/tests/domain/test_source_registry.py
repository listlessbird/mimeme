from __future__ import annotations

import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from tests.factories import (
    create_image,
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)

from mimeme.db.schema import (
    DuplicateReason,
    IngestionSource,
    ProcessingStatus,
    SourceRun,
    SourceRunStatus,
    SourceRunTrigger,
)
from mimeme.domain.source_registry import (
    DuplicateSourceNameError,
    SourceNotFoundError,
    SourceRegistry,
    UnknownAdapterKeyError,
)
from mimeme.domain.source_schedule_spec import DesiredScheduleState

pytestmark = pytest.mark.usefixtures("_patch_async_domain_session_scope")


def _at(day: int) -> datetime.datetime:
    return datetime.datetime(2026, 1, day, tzinfo=datetime.UTC)


async def test_create_persists_source_and_returns_view(async_db_session: AsyncSession) -> None:
    view = await SourceRegistry().create(
        name="r/memes daily",
        adapter_key="meme_api",
        adapter_config={"subreddits": ["memes", "dankmemes"]},
        dataset="memes",
        schedule_cron="0 * * * *",
        schedule_timezone="UTC",
        max_items_per_run=50,
    )

    row = await async_db_session.get(IngestionSource, view.id)
    assert row is not None
    assert row.name == "r/memes daily"
    assert row.adapter_config == {"subreddits": ["memes", "dankmemes"]}
    assert view.enabled is True
    assert view.dataset == "memes"


async def test_create_validates_adapter_and_live_name(async_db_session: AsyncSession) -> None:
    registry = SourceRegistry()
    with pytest.raises(UnknownAdapterKeyError):
        await registry.create(
            name="bad-source",
            adapter_key="not_a_real_adapter",
            adapter_config={},
            dataset=None,
            schedule_cron=None,
            schedule_timezone="UTC",
            max_items_per_run=None,
        )

    assert (
        await async_db_session.scalar(
            select(IngestionSource).where(IngestionSource.name == "bad-source")
        )
        is None
    )

    await registry.create(
        name="dupe",
        adapter_key="meme_api",
        adapter_config={},
        dataset=None,
        schedule_cron=None,
        schedule_timezone="UTC",
        max_items_per_run=None,
    )
    with pytest.raises(DuplicateSourceNameError):
        await registry.create(
            name="dupe",
            adapter_key="meme_api",
            adapter_config={},
            dataset=None,
            schedule_cron=None,
            schedule_timezone="UTC",
            max_items_per_run=None,
        )


async def test_create_reuses_soft_deleted_name() -> None:
    registry = SourceRegistry()
    first = await registry.create(
        name="reuse-me",
        adapter_key="meme_api",
        adapter_config={},
        dataset=None,
        schedule_cron=None,
        schedule_timezone="UTC",
        max_items_per_run=None,
    )
    await registry.soft_delete(first.id)

    second = await registry.create(
        name="reuse-me",
        adapter_key="meme_api",
        adapter_config={},
        dataset=None,
        schedule_cron=None,
        schedule_timezone="UTC",
        max_items_per_run=None,
    )

    assert second.id != first.id


async def test_list_excludes_deleted_and_derives_stats(run_sync_seed) -> None:
    def seed(session: Session) -> int:
        source = create_ingestion_source(session=session, name="stats-src")
        deleted = create_ingestion_source(session=session, name="gone")
        deleted.deleted_at = datetime.datetime.now(datetime.UTC)
        create_source_run(session=session, source=source)
        create_source_run(session=session, source=source)
        for _ in range(3):
            create_source_item(session=session, source=source)
        job = create_job(session=session)
        image = create_image(session=session)
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            duplicate_reason=DuplicateReason.SHA256,
        )
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            status=ProcessingStatus.DONE,
            image_id=image.id,
        )
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            status=ProcessingStatus.FAILED,
        )
        return source.id

    source_id = await run_sync_seed(seed)
    items = await SourceRegistry().list_sources()

    assert [item.id for item in items] == [source_id]
    assert items[0].stats.model_dump() == {
        "run_count": 2,
        "items_discovered": 3,
        "duplicate_count": 1,
        "images_ingested": 1,
        "failed_count": 1,
    }


async def test_list_stats_are_zero_for_fresh_source(run_sync_seed) -> None:
    source_id = await run_sync_seed(
        lambda session: create_ingestion_source(session=session, name="fresh").id
    )

    item = next(item for item in await SourceRegistry().list_sources() if item.id == source_id)

    assert item.stats.model_dump() == {
        "run_count": 0,
        "items_discovered": 0,
        "duplicate_count": 0,
        "images_ingested": 0,
        "failed_count": 0,
    }


async def test_get_returns_recent_runs_with_accounting(run_sync_seed) -> None:
    def seed(session: Session) -> tuple[int, int]:
        source = create_ingestion_source(session=session)
        create_source_run(session=session, source=source, created_at=_at(1))
        run = create_source_run(
            session=session,
            source=source,
            created_at=_at(3),
            status=SourceRunStatus.RUNNING,
            trigger_mode=SourceRunTrigger.MANUAL,
        )
        create_source_run(session=session, source=source, created_at=_at(2))
        job = create_job(session=session)
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            status=ProcessingStatus.DONE,
        )
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            status=ProcessingStatus.DONE,
            duplicate_reason=DuplicateReason.PHASH,
        )
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            status=ProcessingStatus.FAILED,
        )
        for _ in range(4):
            create_source_item(session=session, source=source, last_source_run_id=run.id)
        return source.id, run.id

    source_id, run_id = await run_sync_seed(seed)
    detail = await SourceRegistry().get_source(source_id, recent_runs_limit=2)

    assert [run.created_at for run in detail.recent_runs] == [_at(3), _at(2)]
    run = detail.recent_runs[0]
    assert run.id == run_id
    assert run.status == SourceRunStatus.RUNNING
    assert (run.discovered, run.queued, run.duplicate, run.failed) == (4, 3, 1, 1)


async def test_get_respects_recent_runs_limit(run_sync_seed) -> None:
    def seed(session: Session) -> int:
        source = create_ingestion_source(session=session)
        for day in range(1, 4):
            create_source_run(session=session, source=source, created_at=_at(day))
        return source.id

    source_id = await run_sync_seed(seed)
    detail = await SourceRegistry().get_source(source_id, recent_runs_limit=2)

    assert [run.created_at for run in detail.recent_runs] == [_at(3), _at(2)]


async def test_get_missing_or_deleted_source_raises(run_sync_seed) -> None:
    source_id = await run_sync_seed(lambda session: create_ingestion_source(session=session).id)
    registry = SourceRegistry()
    await registry.soft_delete(source_id)

    with pytest.raises(SourceNotFoundError):
        await registry.get_source(source_id)
    with pytest.raises(SourceNotFoundError):
        await registry.get_source(999_999)


async def test_patch_updates_only_provided_fields(run_sync_seed) -> None:
    source_id = await run_sync_seed(
        lambda session: (
            create_ingestion_source(
                session=session,
                schedule_cron="0 * * * *",
                dataset="memes",
                max_items_per_run=50,
            ).id
        )
    )

    updated = await SourceRegistry().patch(source_id, schedule_cron="0 0 * * *", enabled=False)

    assert updated.schedule_cron == "0 0 * * *"
    assert updated.enabled is False
    assert updated.dataset == "memes"
    assert updated.max_items_per_run == 50


async def test_patch_can_disable_then_reenable(run_sync_seed) -> None:
    source_id = await run_sync_seed(
        lambda session: create_ingestion_source(session=session, enabled=True).id
    )
    registry = SourceRegistry()

    assert (await registry.patch(source_id, enabled=False)).enabled is False
    assert (await registry.patch(source_id, enabled=True)).enabled is True


async def test_patch_rejects_soft_deleted_source(run_sync_seed) -> None:
    source_id = await run_sync_seed(lambda session: create_ingestion_source(session=session).id)
    registry = SourceRegistry()
    await registry.soft_delete(source_id)

    with pytest.raises(SourceNotFoundError):
        await registry.patch(source_id, enabled=False)


async def test_patch_and_delete_missing_source_raise() -> None:
    registry = SourceRegistry()
    with pytest.raises(SourceNotFoundError):
        await registry.patch(999_999, enabled=False)
    with pytest.raises(SourceNotFoundError):
        await registry.soft_delete(999_999)


async def test_soft_delete_preserves_children_and_hides_source(
    run_sync_seed, async_db_session: AsyncSession
) -> None:
    def seed(session: Session) -> tuple[int, int]:
        source = create_ingestion_source(session=session, name="to-delete")
        run = create_source_run(session=session, source=source)
        create_source_item(session=session, source=source)
        return source.id, run.id

    source_id, run_id = await run_sync_seed(seed)
    registry = SourceRegistry()
    await registry.soft_delete(source_id)

    row = await async_db_session.get(IngestionSource, source_id)
    assert row is not None and row.deleted_at is not None
    assert await async_db_session.get(SourceRun, run_id) is not None
    assert all(item.id != source_id for item in await registry.list_sources())
    with pytest.raises(SourceNotFoundError):
        await registry.soft_delete(source_id)


async def test_list_schedule_specs_maps_live_sources(run_sync_seed) -> None:
    def seed(session: Session) -> tuple[int, int]:
        active = create_ingestion_source(session=session, schedule_cron="0 * * * *", enabled=True)
        disabled = create_ingestion_source(
            session=session, schedule_cron="0 0 * * *", enabled=False
        )
        deleted = create_ingestion_source(session=session, schedule_cron="0 1 * * *")
        deleted.deleted_at = datetime.datetime.now(datetime.UTC)
        return active.id, disabled.id

    active_id, disabled_id = await run_sync_seed(seed)
    specs = {spec.source_id: spec for spec in await SourceRegistry().list_schedule_specs()}

    assert set(specs) == {active_id, disabled_id}
    assert specs[active_id].desired_state == DesiredScheduleState.ACTIVE
    assert specs[disabled_id].desired_state == DesiredScheduleState.PAUSED
