from __future__ import annotations

import secrets
from enum import StrEnum
from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from mimeme.config import Settings

log = structlog.getLogger()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class ApiKeyRole(StrEnum):
    ADMIN = "admin"
    READONLY = "readonly"


def _resolve_role(settings: Settings, api_key: str | None) -> ApiKeyRole | None:
    if api_key is None:
        return None

    admin_key = settings.http.api_key_admin
    if admin_key is not None and secrets.compare_digest(api_key, admin_key.get_secret_value()):
        return ApiKeyRole.ADMIN

    readonly_key = settings.http.api_key_readonly
    if readonly_key is not None and secrets.compare_digest(
        api_key, readonly_key.get_secret_value()
    ):
        return ApiKeyRole.READONLY

    return None


def require_admin(
    request: Request, api_key: Annotated[str | None, Security(_api_key_header)] = None
) -> ApiKeyRole:
    settings: Settings = request.app.state.env.settings
    if settings.app_env == "development":
        return ApiKeyRole.ADMIN

    role = _resolve_role(settings, api_key)

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
    settings: Settings = request.app.state.env.settings
    if settings.app_env == "development":
        return ApiKeyRole.ADMIN

    role = _resolve_role(settings, api_key)
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
