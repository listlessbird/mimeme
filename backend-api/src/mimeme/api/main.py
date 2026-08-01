from __future__ import annotations

from typing import cast

import uvicorn
from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ExceptionHandler

from mimeme.api.lifespan import lifespan
from mimeme.api.middleware import register_middleware
from mimeme.api.rate_limit import limiter, rate_limit_exceeded_handler
from mimeme.api.routers import health, images, ingestion, jobs, sources
from mimeme.search import router as search
from mimeme.shared.config import Settings


def create_app(settings: Settings) -> FastAPI:
    limiter.enabled = settings.http.rate_limit_enabled

    middleware = [
        Middleware(SlowAPIMiddleware),
        Middleware(
            CORSMiddleware,  # ty:ignore[invalid-argument-type]
            allow_origins=["*"] if settings.debug else [],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
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

    app.state.settings = settings
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded, cast(ExceptionHandler, rate_limit_exceeded_handler)
    )

    register_middleware(app)

    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(images.router)
    app.include_router(jobs.router)
    app.include_router(sources.router)
    app.include_router(ingestion.router)

    return app


app = create_app(Settings())


def run() -> None:
    settings = Settings()
    uvicorn.run(
        "mimeme.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.logging.level.lower(),
    )


if __name__ == "__main__":
    run()
