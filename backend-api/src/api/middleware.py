from __future__ import annotations

import time

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_SKIP_TIMING_PATHS = {"/live", "/ready", "/health"}


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_timing_middleware(request: Request, call_next):
        if request.url.path in _SKIP_TIMING_PATHS:
            return await call_next(request)
        started = time.monotonic()
        response = await call_next(request)
        structlog.get_logger().info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log = structlog.get_logger()
        log.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
