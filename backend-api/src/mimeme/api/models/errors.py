from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApiErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error detail")


class RateLimitErrorResponse(ApiErrorResponse):
    retry_after: str = Field(description="Rate limit retry hint")


_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ApiErrorResponse, "description": "Bad request"},
    403: {"model": ApiErrorResponse, "description": "Forbidden"},
    404: {"model": ApiErrorResponse, "description": "Not found"},
    409: {"model": ApiErrorResponse, "description": "Conflict"},
    429: {"model": RateLimitErrorResponse, "description": "Rate limit exceeded"},
    500: {"model": ApiErrorResponse, "description": "Internal server error"},
    503: {"model": ApiErrorResponse, "description": "Service unavailable"},
}


def error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    return {status_code: _ERROR_RESPONSES[status_code] for status_code in status_codes}
