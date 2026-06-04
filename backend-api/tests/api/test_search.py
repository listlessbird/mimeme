"""Tests for the /search endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.models.search import SearchResult
from domain.search_index import (
    SearchImageNotFoundError,
    SearchIndexPage,
    SearchIndexUnavailableError,
)


def _make_search_result(
    *,
    id: int = 1,
    sha256: str = "abc123",
    score: float = 0.95,
    url: str | None = "https://mock-s3/presigned",
    caption: str | None = "A meme",
    ocr_text: str | None = None,
    width: int | None = 800,
    height: int | None = 600,
) -> SearchResult:
    return SearchResult(
        id=id,
        sha256=sha256,
        score=score,
        url=url,
        caption=caption,
        ocr_text=ocr_text,
        width=width,
        height=height,
    )


class TestSearchEndpoint:
    def test_search_returns_results(
        self, client: TestClient, mock_index_manager: MagicMock
    ) -> None:
        results = [_make_search_result(id=1), _make_search_result(id=2)]
        page = SearchIndexPage(
            query="funny cat",
            results=results,
            total=2,
            limit=20,
            offset=0,
            search_time_ms=1.0,
            index_version="v1-test",
        )
        with patch("api.routers.search.SearchIndexExecution") as execution_cls:
            execution_cls.return_value.search.return_value = page

            resp = client.get("/search?q=funny+cat")

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "funny cat"
        assert len(data["results"]) == 2
        assert "search_time_ms" in data

    def test_search_respects_limit(self, client: TestClient, mock_index_manager: MagicMock) -> None:
        results = [_make_search_result(id=i) for i in range(3)]
        page = SearchIndexPage(
            query="test",
            results=results,
            total=10,
            limit=3,
            offset=0,
            search_time_ms=1.0,
            index_version="v1-test",
        )
        with patch("api.routers.search.SearchIndexExecution") as execution_cls:
            execution_cls.return_value.search.return_value = page

            resp = client.get("/search?q=test&limit=3")

        data = resp.json()
        assert len(data["results"]) == 3
        assert data["limit"] == 3

    def test_search_empty_query_returns_422(self, client: TestClient) -> None:
        resp = client.get("/search?q=")
        assert resp.status_code == 422

    def test_search_missing_query_returns_422(self, client: TestClient) -> None:
        resp = client.get("/search")
        assert resp.status_code == 422

    def test_search_no_index_returns_503(
        self, client: TestClient, mock_index_manager: MagicMock
    ) -> None:
        """When no index is loaded, search should return 503."""
        with patch("api.routers.search.SearchIndexExecution") as execution_cls:
            execution_cls.return_value.search.side_effect = SearchIndexUnavailableError(
                "Search index not loaded"
            )
            resp = client.get("/search?q=test")
        assert resp.status_code == 503


class TestSimilarEndpoint:
    def test_find_similar_returns_results(
        self, client: TestClient, mock_index_manager: MagicMock
    ) -> None:
        results = [_make_search_result(id=2), _make_search_result(id=3)]
        page = SearchIndexPage(
            query="similar_to:1",
            results=results,
            total=2,
            limit=20,
            offset=0,
            search_time_ms=1.0,
            index_version="v1-test",
        )
        with patch("api.routers.search.SearchIndexExecution") as execution_cls:
            execution_cls.return_value.find_similar.return_value = page
            resp = client.get("/search/similar/1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "similar_to:1"
        assert len(data["results"]) == 2

    def test_find_similar_excludes_query_image(
        self, client: TestClient, mock_index_manager: MagicMock
    ) -> None:
        """Results should not include the query image itself."""
        results = [
            _make_search_result(id=2),
            _make_search_result(id=3),
        ]
        page = SearchIndexPage(
            query="similar_to:1",
            results=results,
            total=2,
            limit=20,
            offset=0,
            search_time_ms=1.0,
            index_version="v1-test",
        )
        with patch("api.routers.search.SearchIndexExecution") as execution_cls:
            execution_cls.return_value.find_similar.return_value = page
            resp = client.get("/search/similar/1")

        data = resp.json()
        result_ids = [r["id"] for r in data["results"]]
        assert 1 not in result_ids

    def test_find_similar_unknown_image_returns_404(
        self, client: TestClient, mock_index_manager: MagicMock
    ) -> None:
        with patch("api.routers.search.SearchIndexExecution") as execution_cls:
            execution_cls.return_value.find_similar.side_effect = SearchImageNotFoundError(
                "Image not in index"
            )
            resp = client.get("/search/similar/999999")
        assert resp.status_code == 404
