from __future__ import annotations

import secrets
from enum import StrEnum
from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from shared.config import settings

log = structlog.getLogger()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class ApiKeyRole(StrEnum):
    ADMIN = "admin"
    READONLY = "readonly"


def _resolve_role(api_key: str | None) -> ApiKeyRole | None:
    if api_key is None:
        return None

    if settings.api_key_admin and secrets.compare_digest(api_key, settings.api_key_admin):
        return ApiKeyRole.ADMIN

    if settings.api_key_readonly and secrets.compare_digest(api_key, settings.api_key_readonly):
        return ApiKeyRole.READONLY

    return None


def require_admin(
    request: Request, api_key: Annotated[str | None, Security(_api_key_header)] = None
) -> ApiKeyRole:
    if settings.app_env == "development":
        return ApiKeyRole.ADMIN

    role = _resolve_role(api_key)

    if role != ApiKeyRole.ADMIN:
        log.warning(
            "auth_denied",
            path=request.url.path,
            method=request.method,
            reason="missing_or_invalid_key",
        )

        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid request")

    return role


def require_readonly(
    request: Request,
    api_key: Annotated[str | None, Security(_api_key_header)] = None,
) -> ApiKeyRole:
    if settings.app_env == "development":
        return ApiKeyRole.ADMIN

    role = _resolve_role(api_key)
    if role is None:
        log.warning(
            "auth_denied",
            path=request.url.path,
            method=request.method,
            reason="missing_or_invalid_key",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Valid API key required",
        )
    return role


AdminRequired = Annotated[ApiKeyRole, Depends(require_admin)]
ReadonlyRequired = Annotated[ApiKeyRole, Depends(require_readonly)]
