from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy.orm import Session
from tests.factories import create_annotation, create_image, create_index_build

import mimeme.domain.search_index as search_index
from mimeme.domain.search_index import (
    SearchEncoderIncompatibleError,
    SearchImageNotFoundError,
    SearchIndexExecution,
    SearchIndexUnavailableError,
    SearchQueryEncodingError,
    SearchService,
    reciprocal_rank_fusion,
)
from mimeme.shared.services.media_url import MediaUrlResolver

MEDIA_URLS = MediaUrlResolver("https://assets.mimeme.dev")


def _execution(index_manager: MagicMock, **kwargs: object) -> SearchIndexExecution:
    return SearchIndexExecution(index_manager, media_urls=MEDIA_URLS, **kwargs)


class _TextEncoder:
    def __init__(self, *, fails: bool = False, source_model: str | None = None) -> None:
        self.fails = fails
        self.queries: list[str] = []
        if source_model is not None:
            self.source_model = source_model

    def encode(self, query: str) -> np.ndarray:
        self.queries.append(query)
        if self.fails:
            raise RuntimeError("encoder failed")
        return np.array([0.1, 0.2], dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_index_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_index, "_last_index_check", 0.0)
    monkeypatch.setattr(search_index, "_active_embed_model", None)
    with search_index._embedding_cache_lock:
        search_index._embedding_cache.clear()


def _patch_read_session_scope(monkeypatch: pytest.MonkeyPatch, db_session: Session) -> None:
    @contextmanager
    def _test_read_session_scope() -> Iterator[Session]:
        yield db_session
        db_session.flush()

    monkeypatch.setattr(search_index, "read_session_scope", _test_read_session_scope)


def test_query_embedding_cache_reuses_normalized_query(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = _TextEncoder()

    def get_instance() -> _TextEncoder:
        return encoder

    monkeypatch.setattr(search_index.SearchTextEncoder, "get_instance", get_instance)
    execution = _execution(
        MagicMock(),
        text_encoder_factory=search_index.SearchTextEncoder.get_instance,
    )

    execution._encode_query("Cat Meme ", "image")
    execution._encode_query("cat meme", "image")

    assert encoder.queries == ["Cat Meme "]


def test_query_embedding_cache_misses_for_different_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _TextEncoder()

    def get_instance() -> _TextEncoder:
        return encoder

    monkeypatch.setattr(search_index.SearchTextEncoder, "get_instance", get_instance)
    execution = _execution(
        MagicMock(),
        text_encoder_factory=search_index.SearchTextEncoder.get_instance,
    )

    execution._encode_query("cat meme", "image")
    execution._encode_query("dog meme", "image")

    assert encoder.queries == ["cat meme", "dog meme"]


def test_query_embedding_cache_evicts_oldest_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = _TextEncoder()

    def get_instance() -> _TextEncoder:
        return encoder

    monkeypatch.setattr(search_index.SearchTextEncoder, "get_instance", get_instance)
    monkeypatch.setattr(search_index, "_EMBEDDING_CACHE_MAX", 1)
    execution = _execution(
        MagicMock(),
        text_encoder_factory=search_index.SearchTextEncoder.get_instance,
    )

    execution._encode_query("first", "image")
    execution._encode_query("second", "image")
    execution._encode_query("first", "image")

    assert encoder.queries == ["first", "second", "first"]


def test_no_active_index_raises_unavailable(
    db_session: Session,
    mock_index_manager: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_read_session_scope(monkeypatch, db_session)
    mock_index_manager.is_loaded = False

    execution = _execution(
        mock_index_manager,
        text_encoder_factory=lambda: _TextEncoder(),
    )

    with pytest.raises(SearchIndexUnavailableError):
        execution.search(query="cat", limit=20, offset=0, mode="image")


def test_stale_loaded_index_triggers_active_index_load(
    db_session: Session,
    mock_index_manager: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_read_session_scope(monkeypatch, db_session)
    create_index_build(session=db_session, version="v2", is_active=True)
    db_session.flush()
    mock_index_manager.is_loaded = True
    mock_index_manager.active_version = "v1"

    _execution(mock_index_manager).ensure_index_loaded_for_thread()

    mock_index_manager.load_active_index.assert_called_once()


def test_query_encoding_failure_is_domain_error(
    db_session: Session,
    mock_index_manager: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_read_session_scope(monkeypatch, db_session)
    create_index_build(session=db_session, version="v1-test", is_active=True)
    db_session.flush()
    mock_index_manager.is_loaded = True
    mock_index_manager.active_version = "v1-test"

    execution = _execution(
        mock_index_manager,
        text_encoder_factory=lambda: _TextEncoder(fails=True),
    )

    with pytest.raises(SearchQueryEncodingError):
        execution.search(query="cat", limit=20, offset=0, mode="image")


def test_encoder_index_model_mismatch_raises_incompatible(
    db_session: Session,
    mock_index_manager: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_read_session_scope(monkeypatch, db_session)
    create_index_build(
        session=db_session,
        version="v1-test",
        is_active=True,
        embed_model="google/siglip2-base-patch16-naflex",
    )
    db_session.flush()
    mock_index_manager.is_loaded = True
    mock_index_manager.active_version = "v1-test"

    execution = _execution(
        mock_index_manager,
        text_encoder_factory=lambda: _TextEncoder(source_model="some/other-model"),
    )

    with pytest.raises(SearchEncoderIncompatibleError):
        execution.search(query="cat", limit=20, offset=0, mode="image")


def test_encoder_index_model_match_searches_normally(
    db_session: Session,
    mock_index_manager: MagicMock,
    mock_storage: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_read_session_scope(monkeypatch, db_session)
    image = create_image(session=db_session)
    create_index_build(
        session=db_session,
        version="v1-test",
        is_active=True,
        embed_model="google/siglip2-base-patch16-naflex",
    )
    db_session.flush()
    mock_index_manager.is_loaded = True
    mock_index_manager.active_version = "v1-test"
    mock_index_manager.has_text_index.return_value = False
    mock_index_manager.search.return_value = [(image.id, 0.9)]

    page = _execution(
        mock_index_manager,
        text_encoder_factory=lambda: _TextEncoder(
            source_model="google/siglip2-base-patch16-naflex"
        ),
    ).search(query="cat", limit=20, offset=0, mode="image")

    assert [result.id for result in page.results] == [image.id]


def test_encoder_without_source_model_skips_guard(
    db_session: Session,
    mock_index_manager: MagicMock,
    mock_storage: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_read_session_scope(monkeypatch, db_session)
    image = create_image(session=db_session)
    create_index_build(
        session=db_session,
        version="v1-test",
        is_active=True,
        embed_model="google/siglip2-base-patch16-naflex",
    )
    db_session.flush()
    mock_index_manager.is_loaded = True
    mock_index_manager.active_version = "v1-test"
    mock_index_manager.has_text_index.return_value = False
    mock_index_manager.search.return_value = [(image.id, 0.9)]

    page = _execution(
        mock_index_manager,
        text_encoder_factory=lambda: _TextEncoder(),
    ).search(query="cat", limit=20, offset=0, mode="image")

    assert [result.id for result in page.results] == [image.id]


def test_search_hydrates_results_and_paginates(
    db_session: Session,
    mock_index_manager: MagicMock,
    mock_storage: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_read_session_scope(monkeypatch, db_session)
    first = create_image(session=db_session)
    second = create_image(session=db_session)
    create_annotation(session=db_session, image=second, caption_text="Second", ocr_text="LOL")
    create_index_build(session=db_session, version="v1-test", is_active=True)
    db_session.flush()
    mock_index_manager.is_loaded = True
    mock_index_manager.active_version = "v1-test"
    mock_index_manager.has_text_index.return_value = False
    mock_index_manager.search.return_value = [(first.id, 0.9), (second.id, 0.8)]

    page = _execution(
        mock_index_manager,
        text_encoder_factory=lambda: _TextEncoder(),
    ).search(query="cat", limit=1, offset=1, mode="image")

    assert page.total == 2
    assert [result.id for result in page.results] == [second.id]
    assert page.results[0].caption == "Second"
    mock_index_manager.search.assert_called_once()
    assert mock_index_manager.search.call_args.kwargs["k"] == 2


def test_hybrid_search_uses_reciprocal_rank_fusion(
    db_session: Session,
    mock_index_manager: MagicMock,
    mock_storage: MagicMock,
) -> None:
    first = create_image(session=db_session)
    second = create_image(session=db_session)
    third = create_image(session=db_session)
    db_session.flush()
    mock_index_manager.is_loaded = True
    mock_index_manager.active_version = "v1-test"
    mock_index_manager.is_text_loaded = False
    mock_index_manager.has_text_index.return_value = True
    mock_index_manager.search.return_value = [(first.id, 0.9), (second.id, 0.8)]
    mock_index_manager.search_text.return_value = [(second.id, 0.95), (third.id, 0.4)]

    results = SearchService(mock_index_manager, MEDIA_URLS).search_by_embedding(
        embedding=[0.1, 0.2],
        db=db_session,
        limit=3,
        mode="hybrid",
    )

    assert [result.id for result in results] == [second.id, first.id, third.id]


def test_hybrid_search_falls_back_to_image_index(
    db_session: Session,
    mock_index_manager: MagicMock,
    mock_storage: MagicMock,
) -> None:
    image = create_image(session=db_session)
    db_session.flush()
    mock_index_manager.is_loaded = True
    mock_index_manager.active_version = "v1-test"
    mock_index_manager.is_text_loaded = False
    mock_index_manager.has_text_index.return_value = False
    mock_index_manager.search.return_value = [(image.id, 0.9)]

    results = SearchService(mock_index_manager, MEDIA_URLS).search_by_embedding(
        embedding=[0.1, 0.2],
        db=db_session,
        limit=3,
        mode="hybrid",
    )

    assert [result.id for result in results] == [image.id]
    mock_index_manager.search_text.assert_not_called()


def test_hydration_skips_missing_rows(
    db_session: Session,
    mock_index_manager: MagicMock,
    mock_storage: MagicMock,
) -> None:
    image = create_image(session=db_session)
    db_session.flush()
    mock_index_manager.is_loaded = True
    mock_index_manager.has_text_index.return_value = False
    mock_index_manager.search.return_value = [(999999, 0.95), (image.id, 0.9)]

    results = SearchService(mock_index_manager, MEDIA_URLS).search_by_embedding(
        embedding=[0.1, 0.2],
        db=db_session,
        limit=2,
        mode="image",
    )

    assert [result.id for result in results] == [image.id]


def test_find_similar_excludes_query_image(
    db_session: Session,
    mock_index_manager: MagicMock,
    mock_storage: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_read_session_scope(monkeypatch, db_session)
    query = create_image(session=db_session)
    second = create_image(session=db_session)
    third = create_image(session=db_session)
    create_index_build(session=db_session, version="v1-test", is_active=True)
    db_session.flush()
    mock_index_manager.is_loaded = True
    mock_index_manager.active_version = "v1-test"
    mock_index_manager.get_vector_by_image_id.return_value = np.array([0.1, 0.2])
    mock_index_manager.search.return_value = [(query.id, 1.0), (second.id, 0.9), (third.id, 0.8)]

    page = _execution(mock_index_manager).find_similar(
        image_id=query.id,
        limit=2,
    )

    assert [result.id for result in page.results] == [second.id, third.id]
    assert mock_index_manager.search.call_args.kwargs["k"] == 3


def test_find_similar_missing_vector_raises_domain_not_found(
    db_session: Session,
    mock_index_manager: MagicMock,
    mock_storage: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_read_session_scope(monkeypatch, db_session)
    create_index_build(session=db_session, version="v1-test", is_active=True)
    db_session.flush()
    mock_index_manager.is_loaded = True
    mock_index_manager.active_version = "v1-test"
    mock_index_manager.get_vector_by_image_id.return_value = None

    with pytest.raises(SearchImageNotFoundError):
        _execution(mock_index_manager).find_similar(
            image_id=999999,
            limit=2,
        )


def test_reciprocal_rank_fusion_orders_shared_results_first() -> None:
    fused = reciprocal_rank_fusion(
        image_results=[(1, 0.9), (2, 0.8)],
        text_results=[(2, 0.95), (3, 0.4)],
    )

    assert [image_id for image_id, _ in fused] == [2, 1, 3]
