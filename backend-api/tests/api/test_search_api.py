from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from api.models.search import SearchResult
from api.routers import search as search_router


class _DummyEncoder:
    def encode(self, _query: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def test_search_returns_503_when_index_unavailable(client, readonly_headers, monkeypatch) -> None:
    def _raise_unavailable(_index_manager) -> None:
        raise HTTPException(status_code=503, detail="Search index not loaded")

    monkeypatch.setattr(search_router, "_ensure_index_loaded_for_thread", _raise_unavailable)
    response = client.get("/search", params={"q": "meme"}, headers=readonly_headers)
    assert response.status_code == 503
    assert response.json()["detail"] == "Search index not loaded"


def test_search_happy_path_supports_pagination(
    client, readonly_headers, fake_index_manager, monkeypatch
) -> None:
    monkeypatch.setattr(search_router, "_ensure_index_loaded_for_thread", lambda _im: None)
    monkeypatch.setattr(search_router, "SearchTextEncoder", type("Enc", (), {"get_instance": _DummyEncoder}))
    
    def _fake_search(_index_manager, _embedding, _limit, _mode="hybrid"):
        return [
            SearchResult(id=11, sha256="a", score=0.9, url="u1", caption="c1", ocr_text="o1"),
            SearchResult(id=12, sha256="b", score=0.8, url="u2", caption="c2", ocr_text="o2"),
        ]

    monkeypatch.setattr(search_router, "_search_by_embedding_for_thread", _fake_search)

    response = client.get(
        "/search",
        params={"q": "cat meme", "limit": 1, "offset": 1},
        headers=readonly_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "cat meme"
    assert body["total"] == 2
    assert body["limit"] == 1
    assert body["offset"] == 1
    assert body["index_version"] == fake_index_manager.active_version
    assert len(body["results"]) == 1
    assert body["results"][0]["id"] == 12


def test_find_similar_excludes_source_image(client, readonly_headers, monkeypatch) -> None:
    monkeypatch.setattr(search_router, "_ensure_index_loaded_for_thread", lambda _im: None)

    def _fake_similar(_index_manager, image_id, _limit):
        return [
            SearchResult(id=image_id, sha256="self", score=1.0),
            SearchResult(id=21, sha256="x1", score=0.9),
            SearchResult(id=22, sha256="x2", score=0.8),
        ]

    monkeypatch.setattr(search_router, "_find_similar_for_thread", _fake_similar)
    response = client.get("/search/similar/20", params={"limit": 5}, headers=readonly_headers)
    assert response.status_code == 200
    result_ids = [item["id"] for item in response.json()["results"]]
    assert 20 not in result_ids
    assert result_ids == [21, 22]


def test_find_similar_returns_404_when_source_missing(client, readonly_headers, monkeypatch) -> None:
    monkeypatch.setattr(search_router, "_ensure_index_loaded_for_thread", lambda _im: None)

    def _missing(*_args, **_kwargs):
        raise ValueError("No embedding found for image ID 999")

    monkeypatch.setattr(search_router, "_find_similar_for_thread", _missing)
    response = client.get("/search/similar/999", headers=readonly_headers)
    assert response.status_code == 404
    assert "No embedding found for image ID 999" in response.json()["detail"]


def test_search_requires_readonly_or_admin_key(client, monkeypatch) -> None:
    monkeypatch.setattr(search_router, "_ensure_index_loaded_for_thread", lambda _im: None)
    monkeypatch.setattr(search_router, "SearchTextEncoder", type("Enc", (), {"get_instance": _DummyEncoder}))
    monkeypatch.setattr(search_router, "_search_by_embedding_for_thread", lambda *_a, **_k: [])

    missing_key = client.get("/search", params={"q": "meme"})
    assert missing_key.status_code == 403


def test_search_validates_query_params(client, readonly_headers) -> None:
    response = client.get("/search", params={"q": ""}, headers=readonly_headers)
    assert response.status_code == 422
