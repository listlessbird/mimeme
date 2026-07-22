from __future__ import annotations

import uuid
from collections.abc import Sequence

from mimeme.db import Db
from mimeme.db.schema import IngestURL
from mimeme.source import rule
from mimeme.source.model import NothingToRetry, RetryPlan
from mimeme.source.store import Store


async def retry_run(db: Db, source_id: int, run_id: int, *, request_id: str | None = None) -> RetryPlan:
    async with db.write_session() as session:
        store = Store(session)
        source = await store.live_source_or_raise(source_id)
        urls = await store.failed_urls_for_run(source_id, run_id)
        return await _reset(store, urls, dataset=source.dataset, request_id=request_id)


async def retry_source(db: Db, source_id: int, *, request_id: str | None = None) -> RetryPlan:
    async with db.write_session() as session:
        store = Store(session)
        source = await store.live_source_or_raise(source_id)
        urls = await store.failed_urls_for_source(source_id)
        return await _reset(store, urls, dataset=source.dataset, request_id=request_id)


async def retry_item(
    db: Db, source_id: int, source_item_id: int, *, request_id: str | None = None
) -> RetryPlan:
    async with db.write_session() as session:
        store = Store(session)
        source = await store.live_source_or_raise(source_id)
        urls = await store.failed_urls_for_item(source_id, source_item_id)
        return await _reset(store, urls, dataset=source.dataset, request_id=request_id)


async def _reset(
    store: Store, urls: Sequence[IngestURL], *, dataset: str | None, request_id: str | None
) -> RetryPlan:
    if not urls:
        raise NothingToRetry

    job_id, run_ids, items = await store.reset_onto_new_job(urls)
    request_id = request_id or uuid.uuid4().hex[:12]
    marker_run_id = run_ids[0] if run_ids else 0
    return RetryPlan(
        job_id=job_id,
        workflow_id=rule.retry_workflow_id(marker_run_id, request_id),
        source_run_ids=run_ids,
        dataset=dataset,
        items=items,
        count=len(urls),
    )
