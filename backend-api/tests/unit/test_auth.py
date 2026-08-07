from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from mimeme.api.auth import ApiKeyRole, _resolve_role, require_admin, require_readonly
from mimeme.config import Settings


def _settings(*, app_env: str = "production", keys: bool = True) -> Settings:
    settings = Settings(app_env=app_env)
    settings.http.api_key_admin = SecretStr("admin-key") if keys else None
    settings.http.api_key_readonly = SecretStr("readonly-key") if keys else None
    return settings


def _request(settings: Settings, *, path: str = "/images", method: str = "POST") -> MagicMock:
    request = MagicMock()
    request.app.state.env = SimpleNamespace(settings=settings)
    request.url.path = path
    request.method = method
    return request


@pytest.mark.parametrize(
    ("key", "role"),
    [
        (None, None),
        ("wrong", None),
        ("admin-key", ApiKeyRole.ADMIN),
        ("readonly-key", ApiKeyRole.READONLY),
    ],
)
def test_resolve_role(key: str | None, role: ApiKeyRole | None) -> None:
    assert _resolve_role(_settings(), key) is role


def test_resolve_role_without_configured_keys() -> None:
    assert _resolve_role(_settings(keys=False), "any") is None


@pytest.mark.parametrize("key", [None, "garbage"])
def test_development_bypasses_admin_auth(key: str | None) -> None:
    assert (
        require_admin(_request(_settings(app_env="development")), api_key=key) is ApiKeyRole.ADMIN
    )


@pytest.mark.parametrize("key", [None, "wrong", "readonly-key"])
def test_admin_rejects_missing_invalid_or_readonly_key(key: str | None) -> None:
    with pytest.raises(HTTPException) as error:
        require_admin(_request(_settings()), api_key=key)
    assert error.value.status_code == 403


def test_admin_accepts_admin_key() -> None:
    assert require_admin(_request(_settings()), api_key="admin-key") is ApiKeyRole.ADMIN


def test_readonly_rejects_missing_key() -> None:
    with pytest.raises(HTTPException) as error:
        require_readonly(_request(_settings(), path="/search", method="GET"), api_key=None)
    assert error.value.status_code == 403


@pytest.mark.parametrize(
    ("key", "role"),
    [("readonly-key", ApiKeyRole.READONLY), ("admin-key", ApiKeyRole.ADMIN)],
)
def test_readonly_accepts_both_configured_roles(key: str, role: ApiKeyRole) -> None:
    assert require_readonly(_request(_settings()), api_key=key) is role
