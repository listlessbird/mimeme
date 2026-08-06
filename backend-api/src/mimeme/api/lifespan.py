from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import structlog
from fastapi import FastAPI

from mimeme import search
from mimeme.config import Settings
from mimeme.env import Env
from mimeme.logging import setup_logging


def _startup_env_snapshot(settings: Settings) -> dict[str, object]:
    return {
        "app_env": settings.app_env,
        "debug": settings.debug,
        "log_level": settings.logging.level,
        "gpu_backend": settings.compute.gpu_backend,
        "embed_model": settings.inference.embed_model,
        "search_encoder_repo": settings.search.encoder_repo,
        "search_encoder_revision": settings.search.encoder_revision,
        "search_encoder_variant": settings.search.encoder_variant,
        "search_encoder_threads": settings.search.encoder_threads,
        "search_hnsw_ef_search": settings.search.hnsw_ef_search,
        "compute_gateway_url": settings.compute.gateway_url,
        "temporal_host": settings.temporal.host,
        "temporal_namespace": settings.temporal.namespace,
        "temporal_task_queue": settings.temporal.task_queue,
        "media_s3_bucket": settings.media.s3_bucket,
        "artifact_s3_bucket": settings.artifacts.s3_bucket,
    }


async def loop_lag_probe(threshold_ms: float = 50.0, interval_s: float = 0.1) -> None:
    log = structlog.get_logger()
    while True:
        started = time.monotonic()
        await asyncio.sleep(interval_s)
        lag_ms = (time.monotonic() - started - interval_s) * 1000
        if lag_ms > threshold_ms:
            log.warning("event_loop_lag", lag_ms=round(lag_ms, 2), threshold_ms=threshold_ms)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    setup_logging(settings, "api")
    log = structlog.get_logger()
    log.info("starting_app", env=settings.app_env)
    log.info("startup_env", **_startup_env_snapshot(settings))

    env = await Env.create(settings)
    app.state.env = env
    for role, store in (("media", env.media), ("artifacts", env.artifacts)):
        try:
            await store.probe()
            log.info("storage_ready", role=role)
        except Exception as exc:
            log.warning("storage_unavailable", role=role, error=str(exc))

    try:
        status = await env.search.status()
        log.info(
            "search_compute_status",
            ready=status.ready,
            serving_version=status.serving_version,
            embed_model=status.embed_model,
            encoder_repo=status.encoder_repo,
            encoder_revision=status.encoder_revision,
            encoder_variant=status.encoder_variant,
        )
    except search.Error as exc:
        log.warning("search_compute_unavailable", error=str(exc))

    if await env.inference.ready():
        log.info("inference_compute_ready")
    else:
        log.warning("inference_compute_unavailable")

    lag_probe = asyncio.create_task(loop_lag_probe(settings.http.loop_lag_threshold_ms))
    try:
        yield
    finally:
        lag_probe.cancel()
        with suppress(asyncio.CancelledError):
            await lag_probe
        await env.aclose()
        log.info("shutting_down_app")
