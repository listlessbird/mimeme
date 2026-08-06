from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mimeme.db.schema import (
    IngestURL,
    ProcessingStatus,
    SourceItem,
    SourceRun,
    SourceRunStatus,
)
from mimeme.source import sync
from mimeme.source.model import (
    DiscoverInput,
    FinishInput,
    RawResponse,
    Retryable,
)
from tests.factories import (
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_run,
)
from tests.job.conftest import SavepointDb
from tests.source.conftest import FakeEnv, FakeHttp, meme_response

MEME_URL = "https://meme-api.com/gimme/memes/50"


def _env(db: SavepointDb, http: FakeHttp | None = None) -> FakeEnv:
    return FakeEnv(db=db, source_http=http or FakeHttp())


async def _source(db: SavepointDb, run_sync_seed, **kwargs) -> int:
    return await run_sync_seed(
        lambda s: (
            create_ingestion_source(
                session=s,
                adapter_config={"subreddits": ["memes"]},
                max_items_per_run=50,
                **kwargs,
            ).id
        )
    )


class TestDiscover:
    async def test_new_items_create_run_job_and_refs(self, db: SavepointDb, run_sync_seed) -> None:
        source_id = await _source(db, run_sync_seed, dataset="d")
        http = FakeHttp()
        http.set(MEME_URL, meme_response("aaa", "bbb"))
        env = _env(db, http)

        result = await sync.discover(env, DiscoverInput(source_id=source_id))

        assert result.discovered == 2 and result.queued == 2
        assert result.ingest_job_id is not None
        assert len(result.items) == 2
        assert result.dataset == "d"

        async with db.read_session() as session:
            run = await session.get(SourceRun, result.source_run_id)
            assert run.status == SourceRunStatus.RUNNING
            assert run.ingest_job_id == result.ingest_job_id
            items = await session.scalar(
                select(func.count(SourceItem.id)).where(SourceItem.source_id == source_id)
            )
            assert items == 2
            urls = (
                await session.scalars(
                    select(IngestURL).where(IngestURL.job_id == result.ingest_job_id)
                )
            ).all()
            assert len(urls) == 2

    async def test_already_seen_items_produce_no_job(self, db: SavepointDb, run_sync_seed) -> None:
        def seed(s: Session) -> int:
            source = create_ingestion_source(
                session=s, adapter_config={"subreddits": ["memes"]}, max_items_per_run=50
            )
            from tests.factories import create_source_item

            create_source_item(session=s, source=source, external_item_id="aaa")
            return source.id

        source_id = await run_sync_seed(seed)
        http = FakeHttp()
        http.set(MEME_URL, meme_response("aaa"))
        env = _env(db, http)

        result = await sync.discover(env, DiscoverInput(source_id=source_id))
        assert result.queued == 0 and result.ingest_job_id is None
        assert result.discovered == 1  # seen, but still discovered

    async def test_terminal_fetch_failure_yields_empty_run(
        self, db: SavepointDb, run_sync_seed
    ) -> None:
        source_id = await _source(db, run_sync_seed)
        http = FakeHttp()
        http.set(MEME_URL, RawResponse(success=False, status_code=404, error="gone"))
        env = _env(db, http)

        result = await sync.discover(env, DiscoverInput(source_id=source_id))
        assert result.discovered == 0 and result.ingest_job_id is None
        async with db.read_session() as session:
            assert await session.get(SourceRun, result.source_run_id) is not None

    async def test_retryable_fetch_raises_before_any_write(
        self, db: SavepointDb, run_sync_seed
    ) -> None:
        source_id = await _source(db, run_sync_seed)
        http = FakeHttp()
        http.set(MEME_URL, Retryable("HTTP 503"))
        env = _env(db, http)

        with pytest.raises(Retryable):
            await sync.discover(env, DiscoverInput(source_id=source_id))

        async with db.read_session() as session:
            runs = await session.scalar(select(func.count(SourceRun.id)))
            assert runs == 0

    async def test_heartbeat_and_cancellation_hooks(self, db: SavepointDb, run_sync_seed) -> None:
        source_id = await _source(db, run_sync_seed)
        http = FakeHttp()
        http.set(MEME_URL, meme_response("aaa"))
        env = _env(db, http)

        beats: list[str] = []
        await sync.discover(
            env,
            DiscoverInput(source_id=source_id),
            heartbeat=beats.append,
            cancelled=lambda: False,
        )
        assert any(b.startswith("fetched:") for b in beats)

    async def test_cancellation_raises(self, db: SavepointDb, run_sync_seed) -> None:
        import asyncio

        source_id = await _source(db, run_sync_seed)
        http = FakeHttp()
        http.set(MEME_URL, meme_response("aaa"))
        env = _env(db, http)

        with pytest.raises(asyncio.CancelledError):
            await sync.discover(env, DiscoverInput(source_id=source_id), cancelled=lambda: True)


class TestFinish:
    async def test_completed_accounting(self, db: SavepointDb, run_sync_seed) -> None:
        def seed(s: Session) -> int:
            source = create_ingestion_source(session=s)
            run = create_source_run(session=s, source=source, status=SourceRunStatus.RUNNING)
            job = create_job(session=s)
            create_ingest_url(
                session=s, job=job, source_run_id=run.id, status=ProcessingStatus.DONE
            )
            return run.id

        run_id = await run_sync_seed(seed)
        result = await sync.finish(_env(db), FinishInput(source_run_id=run_id))
        assert result.status == SourceRunStatus.COMPLETED and result.queued == 1
        async with db.read_session() as session:
            run = await session.get(SourceRun, run_id)
            assert run.status == SourceRunStatus.COMPLETED and run.completed_at is not None

    async def test_partial_accounting(self, db: SavepointDb, run_sync_seed) -> None:
        def seed(s: Session) -> int:
            source = create_ingestion_source(session=s)
            run = create_source_run(session=s, source=source, status=SourceRunStatus.RUNNING)
            job = create_job(session=s)
            create_ingest_url(
                session=s, job=job, source_run_id=run.id, status=ProcessingStatus.DONE
            )
            create_ingest_url(
                session=s, job=job, source_run_id=run.id, status=ProcessingStatus.FAILED
            )
            return run.id

        run_id = await run_sync_seed(seed)
        result = await sync.finish(_env(db), FinishInput(source_run_id=run_id))
        assert result.status == SourceRunStatus.PARTIAL and result.failed == 1

    async def test_error_preserves_original_failure(self, db: SavepointDb, run_sync_seed) -> None:
        def seed(s: Session) -> int:
            source = create_ingestion_source(session=s)
            run = create_source_run(session=s, source=source, status=SourceRunStatus.RUNNING)
            job = create_job(session=s)
            create_ingest_url(
                session=s, job=job, source_run_id=run.id, status=ProcessingStatus.DONE
            )
            return run.id

        run_id = await run_sync_seed(seed)
        result = await sync.finish(
            _env(db), FinishInput(source_run_id=run_id, error="fetch exploded")
        )
        assert result.status == SourceRunStatus.FAILED
        async with db.read_session() as session:
            run = await session.get(SourceRun, run_id)
            assert run.status == SourceRunStatus.FAILED
            assert run.error_message == "fetch exploded"
