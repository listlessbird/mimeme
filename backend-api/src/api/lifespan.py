from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI
from sqlalchemy import select

from api.deps import get_index_manager
from api.services.text_encoder import SearchTextEncoder
from domain.search_index import (
    SearchEncoderIncompatibleError,
    check_encoder_index_compatibility,
)
from shared.config import settings
from shared.db import get_db
from shared.logging import setup_logging
from shared.models import IndexBuild
from shared.services.storage import get_storage_service


def _startup_env_snapshot() -> dict[str, object]:
    return {
        "app_env": settings.app_env,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "gpu_backend": settings.gpu_backend,
        "embed_model": settings.embed_model,
        "embed_device": settings.embed_device,
        "onnx_text_encoder_repo": settings.onnx_text_encoder_repo,
        "onnx_text_encoder_revision": settings.onnx_text_encoder_revision,
        "onnx_text_encoder_variant": settings.onnx_text_encoder_variant,
        "onnx_text_encoder_threads": settings.onnx_text_encoder_threads,
        "preload_text_encoder_on_startup": settings.preload_text_encoder_on_startup,
        "index_type": settings.index_type,
        "index_cache_dir": str(settings.index_cache_dir),
        "temporal_host": settings.temporal_host,
        "temporal_namespace": settings.temporal_namespace,
        "temporal_task_queue": settings.temporal_task_queue,
        "s3_endpoint_url": settings.s3_endpoint_url,
        "s3_region": settings.s3_region,
        "s3_bucket": settings.s3_bucket,
    }


def _index_files_snapshot(version: str | None) -> dict[str, object]:
    if not version:
        return {"cache_version": None}

    cache_path = Path(settings.index_cache_dir) / version
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


def _preload_and_warm_text_encoder() -> None:
    started = time.monotonic()
    encoder = SearchTextEncoder.get_instance()
    encoder.encode("warmup")
    duration_ms = int((time.monotonic() - started) * 1000)
    structlog.get_logger().info("text_encoder_warmed", duration_ms=duration_ms)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log = structlog.get_logger()
    setup_logging("api")

    log.info("starting_app", env=settings.app_env)
    log.info("startup_env", **_startup_env_snapshot())

    settings.index_cache_dir.mkdir(parents=True, exist_ok=True)

    storage = get_storage_service()
    try:
        storage.ensure_bucket_exists()
        log.info("s3_bucket_ready", bucket=storage.bucket)
    except Exception as e:
        log.warning("s3_bucket_init_failed", error=str(e))

    index_manager = get_index_manager()
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
    if settings.preload_text_encoder_on_startup:
        log.info("preloading_text_encoder")
        startup_tasks.append(asyncio.to_thread(_preload_and_warm_text_encoder))

    if startup_tasks:
        results = await asyncio.gather(*startup_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                log.warning("startup_task_failed", error=str(result))

    if settings.preload_text_encoder_on_startup:
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
        **_index_files_snapshot(active_version),
    )

    yield

    log.info("shutting_down_app")
