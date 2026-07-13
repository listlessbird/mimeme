from __future__ import annotations

import asyncio
import time

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.pool import QueuePool

from api.rate_limit import _get_connecting_ip
from shared.config import settings
from shared.db import begin_request_metrics, get_async_engine

_SKIP_TIMING_PATHS = {"/ready", "/health"}


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_timing_middleware(request: Request, call_next):
        if request.url.path in _SKIP_TIMING_PATHS:
            return await call_next(request)

        metrics = begin_request_metrics()
        pool = get_async_engine().pool
        pool_in_use = pool.checkedout() if isinstance(pool, QueuePool) else 0
        started = time.monotonic()
        status_code = 500
        timed_out = False

        try:
            try:
                async with asyncio.timeout(settings.request_timeout_s):
                    response = await call_next(request)
            except TimeoutError:
                timed_out = True
                response = JSONResponse(
                    status_code=504,
                    content={"detail": "Request timed out"},
                )
            status_code = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            structlog.get_logger().info(
                "http_request",
                method=request.method,
                path=request.url.path,
                route=getattr(route, "path", None),
                status_code=status_code,
                duration_ms=int((time.monotonic() - started) * 1000),
                pool_wait_ms=round(metrics.pool_wait_ms, 2),
                db_held_ms=round(metrics.db_held_ms, 2),
                pool_in_use=pool_in_use,
                client_key=_get_connecting_ip(request),
                timed_out=timed_out,
            )

    @app.exception_handler(PoolTimeoutError)
    async def pool_timeout_handler(request: Request, exc: PoolTimeoutError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "Server overloaded, retry shortly"},
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log = structlog.get_logger()
        log.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
