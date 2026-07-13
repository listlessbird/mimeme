from __future__ import annotations

from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

from shared.config import settings


def _get_connecting_ip(request: Request) -> str:
    cf_ip = request.headers.get("CF-Connecting-IP")

    if cf_ip:
        return cf_ip

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    return request.client.host if request.client else "unknown"


SEARCH_LIMIT = "15/minute;100/hour"
ADMIN_LIMIT = "30/minute"
# limit across all clients, prevent dos
GLOBAL_LIMIT = "200/minute"

limiter = Limiter(
    key_func=_get_connecting_ip,
    default_limits=[],
    application_limits=[GLOBAL_LIMIT],
    storage_uri="memory://",
    enabled=settings.rate_limit_enabled,
)


def rate_limit_exceeded_handler(_request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded", "retry_after": exc.detail},
        headers={"Retry-After": str(getattr(exc, "retry_after", 60))},
    )
