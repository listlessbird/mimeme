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
from mimeme.source.fetch import Fetcher
from mimeme.source.fetch import cleanup_checkpoint as cleanup_fetch_checkpoint
from mimeme.source.http import Http
from mimeme.source.model import (
    CleanupInput,
    DiscoveredItem,
    DiscoverInput,
    DiscoverResult,
    FinishInput,
    FinishResult,
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
        replay = (
            await Store(session).replay_discovery(
                source_id=input.source_id, discovery_key=input.checkpoint_id
            )
            if input.checkpoint_id is not None
            else None
        )

    if replay is not None:
        run_id, ingest_job_id, items, discovered_count = replay
        return DiscoverResult(
            source_run_id=run_id,
            ingest_job_id=ingest_job_id,
            dataset=config.dataset,
            items=items,
            discovered=discovered_count,
            queued=len(items),
        )

    adapter = get_adapter(config.adapter_key)
    adapter_config = {**config.adapter_config, "max_items_per_run": config.max_items_per_run}
    if config.adapter_key == "tumblr_tagged":
        api_key = env.settings.tumblr_api_key
        if api_key is None or not api_key.get_secret_value():
            raise RuntimeError("TUMBLR_API_KEY is required for tumblr_tagged sources")
        adapter_config["api_key"] = api_key.get_secret_value()
    discovered_items: list[DiscoveredItem] = []
    fetcher_options = {}
    if config.adapter_key == "kym":
        fetcher_options = {
            key: adapter_config[key]
            for key in ("delay_seconds", "timeout_seconds", "retries", "impersonate")
            if key in adapter_config
        }
    fetcher = Fetcher(
        env.source_http,
        artifacts=getattr(env, "artifacts", None),
        checkpoint_id=input.checkpoint_id,
        **fetcher_options,
    )
    async with fetcher:
        async for item in adapter.discover(adapter_config, fetcher=fetcher, rng=Random()):
            if cancelled is not None and cancelled():
                raise asyncio.CancelledError
            discovered_items.append(item)
            if heartbeat is not None:
                heartbeat(f"discovered:{len(discovered_items)}")

    if heartbeat is not None:
        heartbeat("persisting")

    async with env.db.write_session() as session:
        store = Store(session)
        run_id = await store.create_run(
            source_id=input.source_id,
            trigger=input.trigger,
            discovery_key=input.checkpoint_id,
        )
        media = await store.reconcile_discovery(
            source_id=input.source_id,
            source_run_id=run_id,
            items=discovered_items,
        )

        ingest_job_id: str | None = None
        items = []
        if media:
            ingest_job_id, items = await store.create_ingest_job(
                source_id=input.source_id,
                source_run_id=run_id,
                media=media,
            )
            await store.link_run_ingest_job(source_run_id=run_id, ingest_job_id=ingest_job_id)

    return DiscoverResult(
        source_run_id=run_id,
        ingest_job_id=ingest_job_id,
        dataset=config.dataset,
        items=items,
        discovered=len(discovered_items),
        queued=len(media),
    )


async def cleanup_checkpoint(env: Deps, input: CleanupInput) -> None:
    artifacts = getattr(env, "artifacts", None)
    if artifacts is not None:
        await cleanup_fetch_checkpoint(artifacts, input.checkpoint_id)


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
