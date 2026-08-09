from __future__ import annotations

import asyncio
from collections.abc import Callable
from random import Random
from typing import Protocol

from mimeme.config import Settings
from mimeme.db import Db
from mimeme.db.schema import SourceRunStatus
from mimeme.source import rule
from mimeme.source.adapter import get_adapter
from mimeme.source.http import Http
from mimeme.source.model import (
    DiscoveredItem,
    DiscoverInput,
    DiscoverResult,
    FinishInput,
    FinishResult,
    dedup_source_items,
    derive_run_accounting,
)
from mimeme.source.store import Store

Heartbeat = Callable[[str], None]
Cancelled = Callable[[], bool]


class Deps(Protocol):
    db: Db
    source_http: Http
    settings: Settings


async def discover(
    env: Deps,
    input: DiscoverInput,
    *,
    heartbeat: Heartbeat | None = None,
    cancelled: Cancelled | None = None,
) -> DiscoverResult:
    """Fetch every provider request, then persist discovery, seen items, and the
    ingest job in one transaction. No DB write happens before fetching, so a
    Temporal retry after a transient fetch failure re-runs cleanly."""

    async with env.db.read_session() as session:
        config = await Store(session).live_source_config(input.source_id)

    adapter = get_adapter(config.adapter_key)
    adapter_config = {**config.adapter_config, "max_items_per_run": config.max_items_per_run}
    if config.adapter_key == "tumblr_tagged":
        api_key = env.settings.tumblr_api_key
        if api_key is None or not api_key.get_secret_value():
            raise RuntimeError("TUMBLR_API_KEY is required for tumblr_tagged sources")
        adapter_config["api_key"] = api_key.get_secret_value()
    requests = adapter.build_requests(adapter_config, rng=Random())

    raws = []
    for index, request in enumerate(requests):
        if cancelled is not None and cancelled():
            raise asyncio.CancelledError
        response = await env.source_http.fetch(request)
        if response.success and response.raw is not None:
            raws.append(response.raw)
        if heartbeat is not None:
            heartbeat(f"fetched:{index + 1}/{len(requests)}")

    discovered_items: list[DiscoveredItem] = []
    for raw in raws:
        discovered_items.extend(adapter.parse(raw, adapter_config))

    if heartbeat is not None:
        heartbeat("persisting")

    async with env.db.write_session() as session:
        store = Store(session)
        run_id = await store.create_run(source_id=input.source_id, trigger=input.trigger)
        seen = await store.seen_external_ids(input.source_id)
        dedup = dedup_source_items(discovered_items, seen_ids=seen)
        await store.touch_seen(
            source_id=input.source_id,
            source_run_id=run_id,
            external_ids=[item.external_item_id for item in dedup.already_seen],
        )
        pairs = await store.insert_source_items(
            source_id=input.source_id, source_run_id=run_id, items=dedup.new
        )

        ingest_job_id: str | None = None
        items = []
        if pairs:
            ingest_job_id, items = await store.create_ingest_job(
                source_id=input.source_id, source_run_id=run_id, pairs=pairs
            )
            await store.link_run_ingest_job(source_run_id=run_id, ingest_job_id=ingest_job_id)

    return DiscoverResult(
        source_run_id=run_id,
        ingest_job_id=ingest_job_id,
        dataset=config.dataset,
        items=items,
        discovered=len(dedup.new) + len(dedup.already_seen),
        queued=len(dedup.new),
    )


async def finish(env: Deps, input: FinishInput) -> FinishResult:
    async with env.db.write_session() as session:
        store = Store(session)

        if input.error is not None:
            await store.mark_run_failed(
                source_run_id=input.source_run_id, error=rule.truncate_error(input.error)
            )
            return FinishResult(
                status=SourceRunStatus.FAILED,
                discovered=await store.discovered_count(input.source_run_id),
                queued=0,
                duplicate=0,
                failed=0,
            )

        discovered = await store.discovered_count(input.source_run_id)
        outcomes = await store.url_outcomes(input.source_run_id)
        accounting = derive_run_accounting(discovered_items=discovered, url_outcomes=outcomes)
        await store.set_run_status(source_run_id=input.source_run_id, status=accounting.status)
        return FinishResult(
            status=accounting.status,
            discovered=accounting.discovered,
            queued=accounting.queued,
            duplicate=accounting.duplicate,
            failed=accounting.failed,
        )
