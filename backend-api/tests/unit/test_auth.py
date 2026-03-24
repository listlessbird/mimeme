"""Tests for API authentication and authorization logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from api.auth import ApiKeyRole, _resolve_role, require_admin, require_readonly


# ---------------------------------------------------------------------------
# _resolve_role
# ---------------------------------------------------------------------------


class TestResolveRole:
    def test_none_key_returns_none(self) -> None:
        assert _resolve_role(None) is None

    def test_invalid_key_returns_none(self) -> None:
        with patch("api.auth.settings") as mock_settings:
            mock_settings.api_key_admin = "admin-secret"
            mock_settings.api_key_readonly = "readonly-secret"
            assert _resolve_role("wrong-key") is None

    def test_admin_key_returns_admin(self) -> None:
        with patch("api.auth.settings") as mock_settings:
            mock_settings.api_key_admin = "admin-secret"
            mock_settings.api_key_readonly = "readonly-secret"
            assert _resolve_role("admin-secret") == ApiKeyRole.ADMIN

    def test_readonly_key_returns_readonly(self) -> None:
        with patch("api.auth.settings") as mock_settings:
            mock_settings.api_key_admin = "admin-secret"
            mock_settings.api_key_readonly = "readonly-secret"
            assert _resolve_role("readonly-secret") == ApiKeyRole.READONLY

    def test_no_keys_configured_returns_none(self) -> None:
        with patch("api.auth.settings") as mock_settings:
            mock_settings.api_key_admin = None
            mock_settings.api_key_readonly = None
            assert _resolve_role("any-key") is None


# ---------------------------------------------------------------------------
# require_admin — development mode (auth bypassed)
# ---------------------------------------------------------------------------


class TestRequireAdminDevMode:
    def test_dev_mode_bypasses_auth(self) -> None:
        from unittest.mock import MagicMock

        request = MagicMock()
        with patch("api.auth.settings") as mock_settings:
            mock_settings.app_env = "development"
            role = require_admin(request, api_key=None)
            assert role == ApiKeyRole.ADMIN

    def test_dev_mode_ignores_invalid_key(self) -> None:
        from unittest.mock import MagicMock

        request = MagicMock()
        with patch("api.auth.settings") as mock_settings:
            mock_settings.app_env = "development"
            role = require_admin(request, api_key="garbage")
            assert role == ApiKeyRole.ADMIN


# ---------------------------------------------------------------------------
# require_admin — production mode
# ---------------------------------------------------------------------------


class TestRequireAdminProdMode:
    def test_prod_no_key_raises_403(self) -> None:
        from unittest.mock import MagicMock

        request = MagicMock()
        request.url.path = "/images"
        request.method = "POST"
        with patch("api.auth.settings") as mock_settings:
            mock_settings.app_env = "production"
            mock_settings.api_key_admin = "real-admin-key"
            mock_settings.api_key_readonly = "real-readonly-key"
            with pytest.raises(HTTPException) as exc_info:
                require_admin(request, api_key=None)
            assert exc_info.value.status_code == 403

    def test_prod_wrong_key_raises_403(self) -> None:
        from unittest.mock import MagicMock

        request = MagicMock()
        request.url.path = "/images"
        request.method = "POST"
        with patch("api.auth.settings") as mock_settings:
            mock_settings.app_env = "production"
            mock_settings.api_key_admin = "real-admin-key"
            mock_settings.api_key_readonly = "real-readonly-key"
            with pytest.raises(HTTPException) as exc_info:
                require_admin(request, api_key="wrong-key")
            assert exc_info.value.status_code == 403

    def test_prod_readonly_key_raises_403_for_admin(self) -> None:
        from unittest.mock import MagicMock

        request = MagicMock()
        request.url.path = "/images"
        request.method = "POST"
        with patch("api.auth.settings") as mock_settings:
            mock_settings.app_env = "production"
            mock_settings.api_key_admin = "real-admin-key"
            mock_settings.api_key_readonly = "real-readonly-key"
            with pytest.raises(HTTPException) as exc_info:
                require_admin(request, api_key="real-readonly-key")
            assert exc_info.value.status_code == 403

    def test_prod_valid_admin_key_succeeds(self) -> None:
        from unittest.mock import MagicMock

        request = MagicMock()
        with patch("api.auth.settings") as mock_settings:
            mock_settings.app_env = "production"
            mock_settings.api_key_admin = "real-admin-key"
            mock_settings.api_key_readonly = "real-readonly-key"
            role = require_admin(request, api_key="real-admin-key")
            assert role == ApiKeyRole.ADMIN


# ---------------------------------------------------------------------------
# require_readonly — production mode
# ---------------------------------------------------------------------------


class TestRequireReadonlyProdMode:
    def test_prod_no_key_raises_403(self) -> None:
        from unittest.mock import MagicMock

        request = MagicMock()
        request.url.path = "/search"
        request.method = "GET"
        with patch("api.auth.settings") as mock_settings:
            mock_settings.app_env = "production"
            mock_settings.api_key_admin = "real-admin-key"
            mock_settings.api_key_readonly = "real-readonly-key"
            with pytest.raises(HTTPException) as exc_info:
                require_readonly(request, api_key=None)
            assert exc_info.value.status_code == 403

    def test_prod_readonly_key_succeeds(self) -> None:
        from unittest.mock import MagicMock

        request = MagicMock()
        with patch("api.auth.settings") as mock_settings:
            mock_settings.app_env = "production"
            mock_settings.api_key_admin = "real-admin-key"
            mock_settings.api_key_readonly = "real-readonly-key"
            role = require_readonly(request, api_key="real-readonly-key")
            assert role == ApiKeyRole.READONLY

    def test_prod_admin_key_also_passes_readonly(self) -> None:
        from unittest.mock import MagicMock

        request = MagicMock()
        with patch("api.auth.settings") as mock_settings:
            mock_settings.app_env = "production"
            mock_settings.api_key_admin = "real-admin-key"
            mock_settings.api_key_readonly = "real-readonly-key"
            role = require_readonly(request, api_key="real-admin-key")
            assert role == ApiKeyRole.ADMIN
