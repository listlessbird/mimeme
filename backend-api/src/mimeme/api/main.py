from __future__ import annotations

from typing import cast

import uvicorn
from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.types import ExceptionHandler

from mimeme.api.github_oauth import AuthlibGitHubOAuth
from mimeme.api.lifespan import lifespan
from mimeme.api.middleware import register_middleware
from mimeme.api.rate_limit import limiter, rate_limit_exceeded_handler
from mimeme.api.routers import auth, health, images, ingestion, jobs, search_evals, sources
from mimeme.config import Settings
from mimeme.search import router as search


def create_app(settings: Settings) -> FastAPI:
    limiter.enabled = settings.http.rate_limit_enabled

    debug = settings.debug and settings.app_env != "production"
    cors_origins = list(settings.http.cors_origins)
    if debug:
        cors_origins += [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]

    session_secret = settings.auth.session_secret
    if settings.app_env == "production" and session_secret is None:
        raise ValueError("AUTH_SESSION_SECRET is required in production")
    session_secret_value = (
        session_secret.get_secret_value()
        if session_secret is not None
        else "mimeme-development-session-secret"
    )
    if settings.app_env == "production" and len(session_secret_value) < 32:
        raise ValueError("AUTH_SESSION_SECRET must contain at least 32 characters")

    github_oauth = None
    if settings.auth.github_client_id and settings.auth.github_client_secret is not None:
        github_oauth = AuthlibGitHubOAuth(settings.auth)
    elif settings.app_env == "production":
        raise ValueError("GitHub OAuth credentials are required in production")
    if settings.app_env == "production" and not settings.auth.allowed_github_ids:
        raise ValueError("AUTH_ALLOWED_GITHUB_IDS is required in production")

    middleware = [
        Middleware(SlowAPIMiddleware),
        Middleware(
            SessionMiddleware,
            secret_key=session_secret_value,
            session_cookie=settings.auth.session_cookie,
            max_age=settings.auth.session_max_age_s,
            same_site="lax",
            https_only=settings.app_env == "production",
            domain=settings.auth.cookie_domain,
        ),
        Middleware(
            CORSMiddleware,  # ty:ignore[invalid-argument-type]
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["X-API-Key", "X-Request-ID", "Authorization", "Content-Type"],
        ),
    ]
    app = FastAPI(
        title="Find-Meme API",
        description="Semantic meme search powered by SigLIP embeddings and FAISS",
        version="0.3.0",
        lifespan=lifespan,
        docs_url="/docs" if debug else None,
        redoc_url="/redoc" if debug else None,
        openapi_url="/openapi.json" if debug else None,
        middleware=middleware,
    )

    app.state.settings = settings
    app.state.github_oauth = github_oauth
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded, cast(ExceptionHandler, rate_limit_exceeded_handler)
    )

    register_middleware(app)

    app.include_router(auth.router)
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(images.router)
    app.include_router(jobs.router)
    app.include_router(sources.router)
    app.include_router(ingestion.router)
    app.include_router(search_evals.router)

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
