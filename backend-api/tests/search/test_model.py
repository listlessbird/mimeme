from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from mimeme import search
from mimeme.search.model import PreparedLoad


def test_query_accepts_text_and_similar_searches() -> None:
    text = search.Query(text="funny cat", mode="hybrid", limit=10, offset=3)
    similar = search.Query(similar_image_id=42, limit=5)

    assert text.text == "funny cat"
    assert text.recipe_id == "image_siglip_text"
    assert text.mode == "hybrid"
    assert similar.similar_image_id == 42
    assert similar.mode == "image"


def test_query_preserves_the_public_query_text() -> None:
    assert search.Query(text="  funny cat  ").text == "  funny cat  "


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"text": "cat", "similar_image_id": 42},
        {"text": "   "},
    ],
)
def test_query_requires_exactly_one_non_empty_input(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        search.Query.model_validate(values)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_candidate_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValidationError):
        search.Candidate(image_id=1, score=score)


def test_batch_rejects_duplicate_candidate_ids() -> None:
    with pytest.raises(ValidationError):
        search.Batch(
            candidates=[
                search.Candidate(image_id=1, score=0.9),
                search.Candidate(image_id=1, score=0.8),
            ],
            exhausted=True,
            version="v1",
        )


def test_load_requires_bm25_to_use_its_generation_prefix() -> None:
    files = [
        search.File(name="index.faiss", key="indexes/v2/index.faiss", sha256="0" * 64),
        search.File(name="mapping.json", key="indexes/v2/mapping.json", sha256="1" * 64),
        search.File(name="metadata.json", key="indexes/v2/metadata.json", sha256="2" * 64),
    ]
    encoder = search.Encoder(repo="test/encoder", revision="rev", variant="model.onnx", threads=1)
    bm25 = search.Bm25File(
        key="indexes/other/bm25.sqlite3",
        sha256="3" * 64,
        length=4096,
        count=1,
        weights=(4, 4, 4, 2, 2, 2, 1),
        sqlite_version="3.40.1",
    )

    with pytest.raises(ValidationError, match="BM25 artifact must use its generation prefix"):
        search.Load(version="v2", files=files, bm25=bm25, encoder=encoder)


def test_prepared_load_requires_matching_bm25_descriptor_and_path() -> None:
    encoder = search.Encoder(repo="test/encoder", revision="rev", variant="model.onnx", threads=1)
    paths = {
        "index.faiss": "/tmp/index.faiss",
        "mapping.json": "/tmp/mapping.json",
        "metadata.json": "/tmp/metadata.json",
    }
    bm25 = search.Bm25File(
        key="indexes/v2/bm25.sqlite3",
        sha256="3" * 64,
        length=4096,
        count=1,
        weights=(4, 4, 4, 2, 2, 2, 1),
        sqlite_version="3.40.1",
    )

    with pytest.raises(ValidationError, match="BM25 generation artifact is missing"):
        PreparedLoad(version="v2", workspace="/tmp/v2", paths=paths, bm25=bm25, encoder=encoder)
    with pytest.raises(ValidationError, match="requires a generation descriptor"):
        PreparedLoad(
            version="v2",
            workspace="/tmp/v2",
            paths={**paths, "bm25.sqlite3": "/tmp/bm25.sqlite3"},
            encoder=encoder,
        )
