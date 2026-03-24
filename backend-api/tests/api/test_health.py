"""Tests for the health check endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_live_returns_200_when_healthy(self, client: TestClient) -> None:
        with (
            patch("api.routers.health._check_postgres", return_value=True),
            patch("api.routers.health._check_s3", return_value=True),
            patch("api.routers.health._check_temporal", new_callable=AsyncMock, return_value=True),
        ):
            resp = client.get("/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_live_returns_503_when_postgres_down(self, client: TestClient) -> None:
        with (
            patch("api.routers.health._check_postgres", return_value=False),
            patch("api.routers.health._check_s3", return_value=True),
            patch("api.routers.health._check_temporal", new_callable=AsyncMock, return_value=True),
        ):
            resp = client.get("/live")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"

    def test_live_returns_503_when_s3_down(self, client: TestClient) -> None:
        with (
            patch("api.routers.health._check_postgres", return_value=True),
            patch("api.routers.health._check_s3", return_value=False),
            patch("api.routers.health._check_temporal", new_callable=AsyncMock, return_value=True),
        ):
            resp = client.get("/live")
        assert resp.status_code == 503

    def test_live_returns_503_when_temporal_down(self, client: TestClient) -> None:
        with (
            patch("api.routers.health._check_postgres", return_value=True),
            patch("api.routers.health._check_s3", return_value=True),
            patch("api.routers.health._check_temporal", new_callable=AsyncMock, return_value=False),
        ):
            resp = client.get("/live")
        assert resp.status_code == 503
