from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from api.models.search import SearchResult
from api.routers import search as search_router


class _DummyEncoder:
    def encode(self, _query: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


class _FakeActiveBuild:
    def __init__(self, version: str) -> None:
        self.version = version


class _FakeQuery:
    def __init__(self, active_build: _FakeActiveBuild | None) -> None:
        self._active_build = active_build

    def filter(self, *_args, **_kwargs) -> "_FakeQuery":
        return self

    def first(self) -> _FakeActiveBuild | None:
        return self._active_build


class _FakeDB:
    def __init__(self, active_build: _FakeActiveBuild | None) -> None:
        self._active_build = active_build

    def query(self, _model) -> _FakeQuery:
        return _FakeQuery(self._active_build)


class _FakeIndexManager:
    def __init__(self, is_loaded: bool, active_version: str | None, load_error: Exception | None = None) -> None:
        self.is_loaded = is_loaded
        self.active_version = active_version
        self._load_error = load_error

    def load_active_index(self, _db) -> None:
        if self._load_error is not None:
            raise self._load_error
        self.is_loaded = True


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

    observed_modes: list[str] = []

    def _fake_search(_index_manager, _embedding, _limit, _mode="image"):
        observed_modes.append(_mode)
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
    assert observed_modes == ["image"]


def test_search_hybrid_mode_is_forwarded_when_explicitly_requested(
    client, readonly_headers, monkeypatch
) -> None:
    monkeypatch.setattr(search_router, "_ensure_index_loaded_for_thread", lambda _im: None)
    monkeypatch.setattr(search_router, "SearchTextEncoder", type("Enc", (), {"get_instance": _DummyEncoder}))

    observed_modes: list[str] = []

    def _fake_search(_index_manager, _embedding, _limit, _mode="image"):
        observed_modes.append(_mode)
        return []

    monkeypatch.setattr(search_router, "_search_by_embedding_for_thread", _fake_search)

    response = client.get(
        "/search",
        params={"q": "cat meme", "mode": "hybrid"},
        headers=readonly_headers,
    )
    assert response.status_code == 200
    assert observed_modes == ["hybrid"]


def test_search_rejects_non_hybrid_mode_value(client, readonly_headers) -> None:
    response = client.get(
        "/search",
        params={"q": "cat meme", "mode": "image"},
        headers=readonly_headers,
    )
    assert response.status_code == 422


def test_search_returns_500_when_hybrid_text_search_fails(client, readonly_headers, monkeypatch) -> None:
    monkeypatch.setattr(search_router, "_ensure_index_loaded_for_thread", lambda _im: None)
    monkeypatch.setattr(search_router, "SearchTextEncoder", type("Enc", (), {"get_instance": _DummyEncoder}))

    def _raise_runtime(_index_manager, _embedding, _limit, _mode="hybrid"):
        raise RuntimeError("text faiss failed")

    monkeypatch.setattr(search_router, "_search_by_embedding_for_thread", _raise_runtime)

    response = client.get(
        "/search",
        params={"q": "cat meme", "mode": "hybrid"},
        headers=readonly_headers,
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Search failed"


def test_ensure_index_loaded_returns_503_when_no_active_db_build() -> None:
    db = _FakeDB(active_build=None)
    idx = _FakeIndexManager(is_loaded=True, active_version="v-any")

    with pytest.raises(HTTPException) as exc:
        search_router._ensure_index_loaded(db, idx)

    assert exc.value.status_code == 503
    assert exc.value.detail == "Search index not loaded"


def test_ensure_index_loaded_returns_500_when_reload_raises_generic_error() -> None:
    db = _FakeDB(active_build=_FakeActiveBuild("v-latest"))
    idx = _FakeIndexManager(
        is_loaded=False,
        active_version=None,
        load_error=RuntimeError("disk read failed"),
    )

    with pytest.raises(HTTPException) as exc:
        search_router._ensure_index_loaded(db, idx)

    assert exc.value.status_code == 500
    assert "Failed to load search index: disk read failed" in exc.value.detail


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
