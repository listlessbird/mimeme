from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from api.deps import get_index_manager
from api.rate_limit import limiter, rate_limit_exceeded_handler
from api.routers import health, images, jobs, search
from api.services.text_encoder import SearchTextEncoder
from shared.config import settings
from shared.db import get_db
from shared.services.storage import get_storage_service


def setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            (
                structlog.dev.ConsoleRenderer()
                if settings.debug
                else structlog.processors.JSONRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log = structlog.get_logger()
    setup_logging()

    log.info("starting_app", env=settings.app_env)

    settings.index_cache_dir.mkdir(parents=True, exist_ok=True)

    storage = get_storage_service()
    try:
        storage.ensure_bucket_exists()
        log.info("s3_bucket_ready", bucket=storage.bucket)
    except Exception as e:
        log.warning("s3_bucket_init_failed", error=str(e))

    index_manager = get_index_manager()
    try:
        db_gen = get_db()
        db = next(db_gen)
        try:
            index_manager.load_active_index(db)
            log.info(
                "index_loaded",
                version=index_manager.active_version,
                num_vectors=index_manager.num_vectors,
            )
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
    except FileNotFoundError:
        log.warning("no_index_found", message="Index will be built on first rebuild")
    except Exception as e:
        log.warning("index_load_failed", error=str(e))

    log.info("preloading_text_encoder")
    await asyncio.to_thread(SearchTextEncoder.get_instance)
    log.info("text_encoder_ready")

    yield

    log.info("shutting_down_app")


def create_app() -> FastAPI:
    middleware = [
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

    app.include_router(health.router, tags=["Health"])
    app.include_router(search.router, prefix="/search", tags=["Search"])
    app.include_router(images.router, prefix="/images", tags=["Images"])
    app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])

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
