from __future__ import annotations

import pytest
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

from mimeme.db.schema import IngestURL, ProcessingStatus, SourceRun, SourceRunStatus
from mimeme.domain.source_item_browse import RunNotFoundError
from mimeme.domain.source_registry import SourceNotFoundError
from mimeme.domain.source_retry import NothingToRetryError, SourceItemNotFoundError, SourceRetry

pytestmark = pytest.mark.usefixtures("_patch_async_domain_session_scope")


def _failed_url(session: Session, run, item, **kwargs):
    job = create_job(session=session)
    return create_ingest_url(
        session=session,
        job=job,
        source_id=run.source_id,
        source_run_id=run.id,
        source_item_id=item.id,
        status=ProcessingStatus.FAILED,
        error_message="boom",
        **kwargs,
    )


async def test_retry_run_resets_failed_url_and_queues_job(
    run_sync_seed, async_db_session: AsyncSession
) -> None:
    def seed(session: Session) -> tuple[int, int, int]:
        source = create_ingestion_source(session=session, dataset="memes")
        run = create_source_run(session=session, source=source, status=SourceRunStatus.PARTIAL)
        item = create_source_item(session=session, source=source)
        url = _failed_url(session, run, item)
        return source.id, run.id, url.id

    source_id, run_id, url_id = await run_sync_seed(seed)
    plan = await SourceRetry().retry_run(source_id, run_id)

    url = await async_db_session.get(IngestURL, url_id)
    run = await async_db_session.get(SourceRun, run_id)
    assert url is not None
    assert url.status == ProcessingStatus.PENDING
    assert url.error_message is None
    assert url.job_id == plan.job_id
    assert run is not None and run.status == SourceRunStatus.RUNNING
    assert plan.count == 1
    assert plan.source_run_ids == [run_id]
    assert plan.dataset == "memes"
    assert plan.workflow_id == f"source-retry-workflow-{plan.job_id}"
    assert plan.workflow_id != f"ingest-workflow-{plan.job_id}"


async def test_retry_run_leaves_successful_urls_untouched(
    run_sync_seed, async_db_session: AsyncSession
) -> None:
    def seed(session: Session) -> tuple[int, int, int, str, int]:
        source = create_ingestion_source(session=session)
        run = create_source_run(session=session, source=source)
        failed_item = create_source_item(session=session, source=source)
        done_item = create_source_item(session=session, source=source)
        image = create_image(session=session)
        done_job = create_job(session=session)
        done = create_ingest_url(
            session=session,
            job=done_job,
            source_id=source.id,
            source_run_id=run.id,
            source_item_id=done_item.id,
            status=ProcessingStatus.DONE,
            image_id=image.id,
        )
        failed = _failed_url(session, run, failed_item)
        return source.id, run.id, done.id, done_job.id, failed.id

    source_id, run_id, done_id, done_job_id, failed_id = await run_sync_seed(seed)
    plan = await SourceRetry().retry_run(source_id, run_id)

    done = await async_db_session.get(IngestURL, done_id)
    failed = await async_db_session.get(IngestURL, failed_id)
    assert done is not None and done.status == ProcessingStatus.DONE
    assert done.job_id == done_job_id
    assert failed is not None and failed.job_id == plan.job_id
    assert plan.count == 1


async def test_retry_run_errors(run_sync_seed) -> None:
    def seed(session: Session) -> tuple[int, int]:
        source = create_ingestion_source(session=session)
        run = create_source_run(session=session, source=source)
        return source.id, run.id

    source_id, run_id = await run_sync_seed(seed)
    retry = SourceRetry()
    with pytest.raises(NothingToRetryError):
        await retry.retry_run(source_id, run_id)
    with pytest.raises(RunNotFoundError):
        await retry.retry_run(source_id, 999_999)
    with pytest.raises(SourceNotFoundError):
        await retry.retry_run(999_999, run_id)


async def test_retry_source_resets_failures_across_runs(
    run_sync_seed, async_db_session: AsyncSession
) -> None:
    def seed(session: Session) -> tuple[int, list[int], list[int]]:
        source = create_ingestion_source(session=session, dataset="memes")
        runs = [create_source_run(session=session, source=source) for _ in range(2)]
        items = [create_source_item(session=session, source=source) for _ in range(2)]
        urls = [_failed_url(session, run, item) for run, item in zip(runs, items, strict=True)]
        return source.id, [run.id for run in runs], [url.id for url in urls]

    source_id, run_ids, url_ids = await run_sync_seed(seed)
    plan = await SourceRetry().retry_source(source_id)

    urls = [await async_db_session.get(IngestURL, url_id) for url_id in url_ids]
    assert all(url is not None and url.status == ProcessingStatus.PENDING for url in urls)
    assert all(url is not None and url.job_id == plan.job_id for url in urls)
    assert plan.count == 2
    assert sorted(plan.source_run_ids) == sorted(run_ids)


async def test_retry_source_errors(run_sync_seed) -> None:
    source_id = await run_sync_seed(lambda session: create_ingestion_source(session=session).id)

    with pytest.raises(NothingToRetryError):
        await SourceRetry().retry_source(source_id)
    with pytest.raises(SourceNotFoundError):
        await SourceRetry().retry_source(999_999)


async def test_retry_item_resets_its_failed_attempt(
    run_sync_seed, async_db_session: AsyncSession
) -> None:
    def seed(session: Session) -> tuple[int, int, int, int]:
        source = create_ingestion_source(session=session, dataset="memes")
        run = create_source_run(session=session, source=source)
        item = create_source_item(session=session, source=source)
        url = _failed_url(session, run, item)
        return source.id, run.id, item.id, url.id

    source_id, run_id, item_id, url_id = await run_sync_seed(seed)
    plan = await SourceRetry().retry_item(source_id, item_id)

    url = await async_db_session.get(IngestURL, url_id)
    assert url is not None and url.status == ProcessingStatus.PENDING
    assert url.job_id == plan.job_id
    assert plan.source_run_ids == [run_id]
    assert plan.dataset == "memes"


async def test_retry_item_errors(run_sync_seed) -> None:
    def seed(session: Session) -> tuple[int, int]:
        source = create_ingestion_source(session=session)
        run = create_source_run(session=session, source=source)
        item = create_source_item(session=session, source=source)
        image = create_image(session=session)
        job = create_job(session=session)
        create_ingest_url(
            session=session,
            job=job,
            source_id=source.id,
            source_run_id=run.id,
            source_item_id=item.id,
            status=ProcessingStatus.DONE,
            image_id=image.id,
        )
        return source.id, item.id

    source_id, item_id = await run_sync_seed(seed)
    with pytest.raises(SourceItemNotFoundError):
        await SourceRetry().retry_item(source_id, 999_999)
    with pytest.raises(NothingToRetryError):
        await SourceRetry().retry_item(source_id, item_id)
