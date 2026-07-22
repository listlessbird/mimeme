from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from mimeme.db.schema import (
    IngestionSource,
    IngestURL,
    Job,
    ProcessingStatus,
    SourceRun,
    SourceRunStatus,
    SourceRunTrigger,
)
from mimeme.ingest.model import RemoteUrl
from mimeme.source import store as source_store
from mimeme.source.model import (
    DiscoveredItem,
    DuplicateSourceName,
    RunNotFound,
    SourceItemNotFound,
    SourceNotFound,
)
from mimeme.source.store import Store
from tests.factories import (
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)
from tests.job.conftest import PoolDb, SavepointDb


def _item(ext: str) -> DiscoveredItem:
    return DiscoveredItem(external_item_id=ext, media_url=f"https://a/{ext}.jpg", title=ext)


class TestCrud:
    async def test_create_and_get(self, db: SavepointDb) -> None:
        view = await source_store.create(
            db,
            name="daily",
            adapter_key="meme_api",
            adapter_config={"subreddits": ["memes"]},
            dataset="d",
            schedule_cron="0 * * * *",
            schedule_timezone="UTC",
            max_items_per_run=10,
        )
        detail = await source_store.get_source(db, view.id)
        assert detail.name == "daily" and detail.dataset == "d"
        assert detail.stats.run_count == 0

    async def test_live_name_uniqueness(self, db: SavepointDb) -> None:
        await source_store.create(
            db,
            name="dup",
            adapter_key="meme_api",
            adapter_config={"subreddits": ["memes"]},
            dataset=None,
            schedule_cron=None,
            schedule_timezone="UTC",
            max_items_per_run=None,
        )
        with pytest.raises(DuplicateSourceName):
            await source_store.create(
                db,
                name="dup",
                adapter_key="meme_api",
                adapter_config={"subreddits": ["memes"]},
                dataset=None,
                schedule_cron=None,
                schedule_timezone="UTC",
                max_items_per_run=None,
            )

    async def test_soft_delete_then_recreate_same_name(self, db: SavepointDb) -> None:
        first = await source_store.create(
            db,
            name="reuse",
            adapter_key="meme_api",
            adapter_config={"subreddits": ["memes"]},
            dataset=None,
            schedule_cron=None,
            schedule_timezone="UTC",
            max_items_per_run=None,
        )
        await source_store.soft_delete(db, first.id)
        again = await source_store.create(
            db,
            name="reuse",
            adapter_key="meme_api",
            adapter_config={"subreddits": ["memes"]},
            dataset=None,
            schedule_cron=None,
            schedule_timezone="UTC",
            max_items_per_run=None,
        )
        assert again.id != first.id
        with pytest.raises(SourceNotFound):
            await source_store.get_source(db, first.id)

    async def test_patch_changes_fields(self, db: SavepointDb) -> None:
        view = await source_store.create(
            db,
            name="p",
            adapter_key="meme_api",
            adapter_config={"subreddits": ["memes"]},
            dataset=None,
            schedule_cron=None,
            schedule_timezone="UTC",
            max_items_per_run=None,
        )
        patched = await source_store.patch(db, view.id, {"enabled": False, "schedule_cron": "*/5 * * * *"})
        assert patched.enabled is False and patched.schedule_cron == "*/5 * * * *"


class TestDiscoveryPersistence:
    async def test_insert_items_and_ingest_job_atomic(
        self, db: SavepointDb, run_sync_seed
    ) -> None:
        source_id = await run_sync_seed(lambda s: create_ingestion_source(session=s).id)
        async with db.write_session() as session:
            store = Store(session)
            run_id = await store.create_run(source_id=source_id, trigger=SourceRunTrigger.MANUAL)
            pairs = await store.insert_source_items(
                source_id=source_id, source_run_id=run_id, items=[_item("a"), _item("b")]
            )
            job_id, refs = await store.create_ingest_job(
                source_id=source_id, source_run_id=run_id, pairs=pairs
            )
            await store.link_run_ingest_job(source_run_id=run_id, ingest_job_id=job_id)

        assert len(refs) == 2
        assert all(isinstance(r.source, RemoteUrl) for r in refs)
        async with db.read_session() as session:
            job = await session.get(Job, job_id)
            assert job is not None
            urls = (
                await session.scalars(select(IngestURL).where(IngestURL.job_id == job_id))
            ).all()
            assert len(urls) == 2
            assert {u.id for u in urls} == {r.item_id for r in refs}
            assert all(u.source_run_id == run_id and u.source_id == source_id for u in urls)
            run = await session.get(SourceRun, run_id)
            assert run.ingest_job_id == job_id

    async def test_seen_ids_excludes_existing(self, db: SavepointDb, run_sync_seed) -> None:
        def seed(s: Session) -> int:
            source = create_ingestion_source(session=s)
            create_source_item(session=s, source=source, external_item_id="old")
            return source.id

        source_id = await run_sync_seed(seed)
        async with db.read_session() as session:
            seen = await Store(session).seen_external_ids(source_id)
        assert "old" in seen


class TestAccountingQueries:
    async def test_discovered_count_and_outcomes(self, db: SavepointDb, run_sync_seed) -> None:
        def seed(s: Session) -> tuple[int, int]:
            source = create_ingestion_source(session=s)
            run = create_source_run(session=s, source=source)
            job = create_job(session=s)
            create_source_item(
                session=s, source=source, external_item_id="x", last_source_run_id=run.id
            )
            create_ingest_url(
                session=s,
                job=job,
                source_run_id=run.id,
                status=ProcessingStatus.DONE,
            )
            create_ingest_url(
                session=s,
                job=job,
                source_run_id=run.id,
                status=ProcessingStatus.FAILED,
            )
            return source.id, run.id

        _, run_id = await run_sync_seed(seed)
        async with db.read_session() as session:
            store = Store(session)
            assert await store.discovered_count(run_id) == 1
            outcomes = await store.url_outcomes(run_id)
        statuses = sorted(o.status for o in outcomes)
        assert statuses == [ProcessingStatus.DONE, ProcessingStatus.FAILED]


class TestRetryReset:
    async def test_reset_moves_failed_urls_to_new_job(
        self, db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(s: Session) -> tuple[int, int]:
            source = create_ingestion_source(session=s)
            run = create_source_run(session=s, source=source, status=SourceRunStatus.FAILED)
            job = create_job(session=s)
            create_ingest_url(
                session=s,
                job=job,
                source_id=source.id,
                source_run_id=run.id,
                status=ProcessingStatus.FAILED,
                error_message="boom",
                input_kind="remote_image_url",
                url="https://a/1.jpg",
            )
            return source.id, run.id

        source_id, run_id = await run_sync_seed(seed)
        async with db.write_session() as session:
            store = Store(session)
            urls = await store.failed_urls_for_run(source_id, run_id)
            new_job_id, run_ids, refs = await store.reset_onto_new_job(urls)

        assert run_ids == [run_id] and len(refs) == 1
        async with db.read_session() as session:
            url = (
                await session.scalars(select(IngestURL).where(IngestURL.job_id == new_job_id))
            ).one()
            assert url.status == ProcessingStatus.PENDING and url.error_message is None
            run = await session.get(SourceRun, run_id)
            assert run.status == SourceRunStatus.RUNNING

    async def test_missing_run_raises(self, db: SavepointDb, run_sync_seed) -> None:
        source_id = await run_sync_seed(lambda s: create_ingestion_source(session=s).id)
        async with db.write_session() as session:
            with pytest.raises(RunNotFound):
                await Store(session).failed_urls_for_run(source_id, 999999)

    async def test_missing_item_raises(self, db: SavepointDb, run_sync_seed) -> None:
        source_id = await run_sync_seed(lambda s: create_ingestion_source(session=s).id)
        async with db.write_session() as session:
            with pytest.raises(SourceItemNotFound):
                await Store(session).failed_urls_for_item(source_id, 999999)


class TestConcurrentLiveNameCreate:
    async def test_unique_index_blocks_second_live_name(
        self, pool_db: PoolDb, run_sync_seed
    ) -> None:
        name = "race-name"

        async def racer() -> str:
            try:
                async with pool_db.write_session() as session:
                    store = Store(session)
                    if await store.live_name_exists(name):
                        return "duplicate"
                    await store.insert_source(
                        name=name,
                        adapter_key="meme_api",
                        adapter_config={"subreddits": ["memes"]},
                        dataset=None,
                        schedule_cron=None,
                        schedule_timezone="UTC",
                        max_items_per_run=None,
                        enabled=True,
                    )
                    return "winner"
            except Exception:
                return "conflict"

        results = await asyncio.gather(racer(), racer())
        assert results.count("winner") == 1
        async with pool_db.read_session() as session:
            rows = (
                await session.scalars(select(IngestionSource).where(IngestionSource.name == name))
            ).all()
        assert len(rows) == 1
        # cleanup (pool_db is a real DB, not rolled back)
        async with pool_db.write_session() as session:
            obj = await session.get(IngestionSource, rows[0].id)
            await session.delete(obj)
