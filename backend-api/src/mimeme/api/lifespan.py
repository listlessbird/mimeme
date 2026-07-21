from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI
from sqlalchemy import select

from mimeme.api.services.text_encoder import SearchTextEncoder
from mimeme.db.schema import IndexBuild
from mimeme.domain.search_index import (
    SearchEncoderIncompatibleError,
    check_encoder_index_compatibility,
)
from mimeme.env import Env
from mimeme.shared.config import Settings
from mimeme.shared.db import get_db
from mimeme.shared.logging import setup_logging


def _startup_env_snapshot(settings: Settings) -> dict[str, object]:
    return {
        "app_env": settings.app_env,
        "debug": settings.debug,
        "log_level": settings.logging.level,
        "gpu_backend": settings.compute.gpu_backend,
        "embed_model": settings.inference.embed_model,
        "embed_device": settings.inference.embed_device,
        "onnx_text_encoder_repo": settings.inference.onnx_text_encoder_repo,
        "onnx_text_encoder_revision": settings.inference.onnx_text_encoder_revision,
        "onnx_text_encoder_variant": settings.inference.onnx_text_encoder_variant,
        "onnx_text_encoder_threads": settings.inference.onnx_text_encoder_threads,
        "preload_text_encoder_on_startup": settings.inference.preload_text_encoder_on_startup,
        "index_type": settings.index.type,
        "index_cache_dir": str(settings.index.cache_dir),
        "temporal_host": settings.temporal.host,
        "temporal_namespace": settings.temporal.namespace,
        "temporal_task_queue": settings.temporal.task_queue,
        "media_s3_endpoint_url": settings.media.s3_endpoint_url,
        "media_s3_region": settings.media.s3_region,
        "media_s3_bucket": settings.media.s3_bucket,
        "artifact_s3_endpoint_url": settings.artifacts.s3_endpoint_url,
        "artifact_s3_region": settings.artifacts.s3_region,
        "artifact_s3_bucket": settings.artifacts.s3_bucket,
    }


def _index_files_snapshot(settings: Settings, version: str | None) -> dict[str, object]:
    if not version:
        return {"cache_version": None}

    cache_path = Path(settings.index.cache_dir) / version
    return {
        "cache_version": version,
        "cache_path": str(cache_path),
        "cache_index_exists": (cache_path / "index.faiss").exists(),
        "cache_mapping_exists": (cache_path / "mapping.json").exists(),
        "cache_metadata_exists": (cache_path / "metadata.json").exists(),
        "cache_text_index_exists": (cache_path / "text_index.faiss").exists(),
        "cache_text_mapping_exists": (cache_path / "text_mapping.json").exists(),
        "cache_text_metadata_exists": (cache_path / "text_metadata.json").exists(),
    }


async def loop_lag_probe(threshold_ms: float, interval_s: float = 0.1) -> None:
    log = structlog.get_logger()
    while True:
        started = time.monotonic()
        await asyncio.sleep(interval_s)
        lag_ms = (time.monotonic() - started - interval_s) * 1000
        if lag_ms > threshold_ms:
            log.warning("event_loop_lag", lag_ms=round(lag_ms, 2), threshold_ms=threshold_ms)


def _preload_and_warm_text_encoder() -> None:
    started = time.monotonic()
    encoder = SearchTextEncoder.get_instance()
    encoder.encode("warmup")
    duration_ms = int((time.monotonic() - started) * 1000)
    structlog.get_logger().info("text_encoder_warmed", duration_ms=duration_ms)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    setup_logging("api")
    log = structlog.get_logger()

    log.info("starting_app", env=settings.app_env)
    log.info("startup_env", **_startup_env_snapshot(settings))

    settings.index.cache_dir.mkdir(parents=True, exist_ok=True)

    env = await Env.create(settings)
    app.state.env = env

    for role, storage in (
        ("media", env.media_storage),
        ("artifact", env.artifact_storage),
    ):
        try:
            if await storage.bucket_exists():
                log.info("storage_bucket_ready", role=role, bucket=storage.bucket)
            else:
                log.warning("storage_bucket_unavailable", role=role, bucket=storage.bucket)
        except Exception as e:
            log.warning("storage_bucket_check_failed", role=role, error=str(e))

    index_manager = env.index_manager
    db_active_version: str | None = None
    db_embed_model: str | None = None
    autoloaded_version: str | None = None
    try:
        db_gen = get_db()
        db = next(db_gen)
        try:
            active_build = db.scalars(
                select(IndexBuild).where(IndexBuild.is_active.is_(True))
            ).first()
            db_active_version = active_build.version if active_build else None
            db_embed_model = active_build.embed_model if active_build else None
            if not db_active_version:
                log.warning("no_active_index_in_db")

            autoloaded_version = index_manager.autoload_latest_available(db)
            if autoloaded_version:
                db_active_version = autoloaded_version
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
    except Exception as e:
        log.warning("index_load_failed", error=str(e))

    startup_tasks: list[Any] = []
    if db_active_version:
        startup_tasks.append(asyncio.to_thread(index_manager.load_index_version, db_active_version))
    if settings.inference.preload_text_encoder_on_startup:
        log.info("preloading_text_encoder")
        startup_tasks.append(asyncio.to_thread(_preload_and_warm_text_encoder))

    if startup_tasks:
        results = await asyncio.gather(*startup_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                log.warning("startup_task_failed", error=str(result))

    if settings.inference.preload_text_encoder_on_startup:
        try:
            encoder = await asyncio.to_thread(SearchTextEncoder.get_instance)
            check_encoder_index_compatibility(encoder, db_embed_model)
        except SearchEncoderIncompatibleError:
            pass
        except Exception as e:
            log.warning("text_encoder_preload_failed", error=str(e))

    if index_manager.is_loaded:
        log.info(
            "index_loaded",
            version=index_manager.active_version,
            num_vectors=index_manager.num_vectors,
        )

    if index_manager.is_loaded and index_manager.has_text_index():
        log.info("preloading_text_index")
        await asyncio.to_thread(index_manager.ensure_text_index_loaded)
        log.info("text_index_ready")

    active_version = index_manager.active_version or db_active_version
    log.info(
        "startup_index_status",
        db_active_version=db_active_version,
        autoloaded_version=autoloaded_version,
        manager_is_loaded=index_manager.is_loaded,
        manager_active_version=index_manager.active_version,
        manager_num_vectors=index_manager.num_vectors,
        manager_is_text_loaded=index_manager.is_text_loaded,
        manager_has_text_index=index_manager.has_text_index(),
        **_index_files_snapshot(settings, active_version),
    )

    lag_probe_task = asyncio.create_task(loop_lag_probe(settings.http.loop_lag_threshold_ms))

    try:
        yield
    finally:
        lag_probe_task.cancel()
        with suppress(asyncio.CancelledError):
            await lag_probe_task
        await env.aclose()

    log.info("shutting_down_app")