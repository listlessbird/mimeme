from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from mimeme import search


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
