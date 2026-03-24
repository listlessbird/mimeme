"""Tests for the /search endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.models.search import SearchResult


def _make_search_result(**overrides: object) -> SearchResult:
    defaults = {
        "id": 1,
        "sha256": "abc123",
        "score": 0.95,
        "url": "https://mock-s3/presigned",
        "caption": "A meme",
        "ocr_text": None,
        "width": 800,
        "height": 600,
    }
    defaults.update(overrides)
    return SearchResult(**defaults)


class TestSearchEndpoint:
    def test_search_returns_results(
        self, client: TestClient, mock_index_manager: MagicMock
    ) -> None:
        results = [_make_search_result(id=1), _make_search_result(id=2)]
        with (
            patch("api.routers.search._ensure_index_loaded_for_thread"),
            patch("api.routers.search._search_by_embedding_for_thread", return_value=results),
            patch("api.routers.search.SearchTextEncoder") as mock_encoder_cls,
        ):
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
            mock_encoder_cls.get_instance.return_value = mock_encoder

            resp = client.get("/search?q=funny+cat")

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "funny cat"
        assert len(data["results"]) == 2
        assert "search_time_ms" in data

    def test_search_respects_limit(
        self, client: TestClient, mock_index_manager: MagicMock
    ) -> None:
        results = [_make_search_result(id=i) for i in range(10)]
        with (
            patch("api.routers.search._ensure_index_loaded_for_thread"),
            patch("api.routers.search._search_by_embedding_for_thread", return_value=results),
            patch("api.routers.search.SearchTextEncoder") as mock_encoder_cls,
        ):
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
            mock_encoder_cls.get_instance.return_value = mock_encoder

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
        from fastapi import HTTPException

        def _raise_503(*args, **kwargs):
            raise HTTPException(status_code=503, detail="Search index not loaded")

        with patch("api.routers.search._ensure_index_loaded_for_thread", side_effect=_raise_503):
            resp = client.get("/search?q=test")
        assert resp.status_code == 503


class TestSimilarEndpoint:
    def test_find_similar_returns_results(
        self, client: TestClient, mock_index_manager: MagicMock
    ) -> None:
        results = [_make_search_result(id=2), _make_search_result(id=3)]
        with (
            patch("api.routers.search._ensure_index_loaded_for_thread"),
            patch("api.routers.search._find_similar_for_thread", return_value=results),
        ):
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
            _make_search_result(id=1),  # the query image
            _make_search_result(id=2),
            _make_search_result(id=3),
        ]
        with (
            patch("api.routers.search._ensure_index_loaded_for_thread"),
            patch("api.routers.search._find_similar_for_thread", return_value=results),
        ):
            resp = client.get("/search/similar/1")

        data = resp.json()
        result_ids = [r["id"] for r in data["results"]]
        assert 1 not in result_ids

    def test_find_similar_unknown_image_returns_404(
        self, client: TestClient, mock_index_manager: MagicMock
    ) -> None:
        with (
            patch("api.routers.search._ensure_index_loaded_for_thread"),
            patch(
                "api.routers.search._find_similar_for_thread",
                side_effect=ValueError("Image not in index"),
            ),
        ):
            resp = client.get("/search/similar/999999")
        assert resp.status_code == 404
