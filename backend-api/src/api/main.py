from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from api.deps import get_index_manager
from api.rate_limit import limiter, rate_limit_exceeded_handler
from api.routers import health, images, jobs, search
from api.services.text_encoder import SearchTextEncoder
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
        "search_text_encoder_device": settings.search_text_encoder_device,
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
    autoloaded_version: str | None = None
    try:
        db_gen = get_db()
        db = next(db_gen)
        try:
            active_build = db.query(IndexBuild).filter(IndexBuild.is_active).first()
            db_active_version = active_build.version if active_build else None
            if db_active_version:
                index_manager.load_active_index(db)
                log.info(
                    "index_loaded",
                    version=index_manager.active_version,
                    num_vectors=index_manager.num_vectors,
                )
            else:
                log.warning("no_active_index_in_db")

            autoloaded_version = index_manager.autoload_latest_available(db)
            if autoloaded_version:
                log.info(
                    "index_autoloaded_latest",
                    version=autoloaded_version,
                    num_vectors=index_manager.num_vectors,
                )
                db_active_version = autoloaded_version
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
    except FileNotFoundError:
        log.warning("no_index_found", message="Index will be built on first rebuild")
    except Exception as e:
        log.warning("index_load_failed", error=str(e))

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

    log.info("preloading_text_encoder")
    await asyncio.to_thread(SearchTextEncoder.get_instance)
    log.info("text_encoder_ready")

    yield

    log.info("shutting_down_app")


def create_app() -> FastAPI:
    middleware = [
        Middleware(SlowAPIMiddleware),
        Middleware(
            CORSMiddleware,  # ty:ignore[invalid-argument-type]
            allow_origins=["*"] if settings.debug else [],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
    app = FastAPI(
        title="Find-Meme API",
        description="Semantic meme search powered by SigLIP embeddings and FAISS",
        version="0.3.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        middleware=middleware,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log = structlog.get_logger()
        log.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(images.router)
    app.include_router(jobs.router)

    return app


app = create_app()


def run() -> None:
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
