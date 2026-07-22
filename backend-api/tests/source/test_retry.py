from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from mimeme.db.schema import IngestURL, ProcessingStatus, SourceRunStatus
from mimeme.ingest.model import RemoteUrl
from mimeme.source import retry
from mimeme.source.model import NothingToRetry, RunNotFound, SourceItemNotFound, SourceNotFound
from tests.factories import (
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)
from tests.job.conftest import SavepointDb


def _seed_failed(s: Session):
    source = create_ingestion_source(session=s, dataset="d")
    run = create_source_run(session=s, source=source, status=SourceRunStatus.FAILED)
    item = create_source_item(session=s, source=source, external_item_id="e1")
    job = create_job(session=s)
    url = create_ingest_url(
        session=s,
        job=job,
        source_id=source.id,
        source_run_id=run.id,
        source_item_id=item.id,
        status=ProcessingStatus.FAILED,
        error_message="boom",
        url="https://a/1.jpg",
    )
    return source.id, run.id, item.id, url.id


class TestRetryRun:
    async def test_builds_plan_and_resets_urls(self, db: SavepointDb, run_sync_seed) -> None:
        source_id, run_id, _item_id, _url_id = await run_sync_seed(_seed_failed)

        plan = await retry.retry_run(db, source_id, run_id, request_id="req1")

        assert plan.count == 1 and plan.source_run_ids == [run_id]
        assert plan.dataset == "d"
        assert plan.workflow_id == f"source-retry-v2-{run_id}-req1"
        assert len(plan.items) == 1 and isinstance(plan.items[0].source, RemoteUrl)

        async with db.read_session() as session:
            url = (
                await session.scalars(select(IngestURL).where(IngestURL.job_id == plan.job_id))
            ).one()
            assert url.status == ProcessingStatus.PENDING and url.error_message is None

    async def test_unknown_run_raises(self, db: SavepointDb, run_sync_seed) -> None:
        source_id, _run_id, _i, _u = await run_sync_seed(_seed_failed)
        with pytest.raises(RunNotFound):
            await retry.retry_run(db, source_id, 987654)

    async def test_unknown_source_raises(self, db: SavepointDb) -> None:
        with pytest.raises(SourceNotFound):
            await retry.retry_source(db, 123456)


class TestRetrySource:
    async def test_retries_all_failed(self, db: SavepointDb, run_sync_seed) -> None:
        source_id, run_id, _i, _u = await run_sync_seed(_seed_failed)
        plan = await retry.retry_source(db, source_id)
        assert plan.count == 1 and plan.source_run_ids == [run_id]

    async def test_nothing_to_retry(self, db: SavepointDb, run_sync_seed) -> None:
        source_id = await run_sync_seed(lambda s: create_ingestion_source(session=s).id)
        with pytest.raises(NothingToRetry):
            await retry.retry_source(db, source_id)


class TestRetryItem:
    async def test_retries_item(self, db: SavepointDb, run_sync_seed) -> None:
        source_id, _run_id, item_id, _u = await run_sync_seed(_seed_failed)
        plan = await retry.retry_item(db, source_id, item_id, request_id="r2")
        assert plan.count == 1

    async def test_unknown_item_raises(self, db: SavepointDb, run_sync_seed) -> None:
        source_id, _run_id, _i, _u = await run_sync_seed(_seed_failed)
        with pytest.raises(SourceItemNotFound):
            await retry.retry_item(db, source_id, 987654)
