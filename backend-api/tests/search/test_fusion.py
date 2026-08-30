from __future__ import annotations

import pytest

from mimeme.search import fusion


def test_rrf_accepts_one_two_or_three_rankings() -> None:
    assert [item.image_id for item in fusion.rrf([[3, 1, 2]], k=60)] == [3, 1, 2]
    assert [item.image_id for item in fusion.rrf([[1, 2], [2, 3]], k=60)] == [2, 1, 3]
    assert [item.image_id for item in fusion.rrf([[1, 2], [2, 3], [3, 1]], k=60)] == [1, 2, 3]


def test_rrf_breaks_score_ties_by_best_rank_then_image_id() -> None:
    ranked = fusion.rrf([[4, 2], [3, 2]], k=60)
    assert [item.image_id for item in ranked] == [2, 3, 4]


def test_rrf_rejects_duplicate_ids_within_a_ranking() -> None:
    with pytest.raises(ValueError, match="unique image IDs"):
        fusion.rrf([[1, 1]], k=60)
