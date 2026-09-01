from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Ranked:
    image_id: int
    score: float
    best_rank: int


def rrf(rankings: Sequence[Sequence[int]], *, k: int) -> list[Ranked]:
    if not rankings:
        raise ValueError("RRF requires at least one ranking")
    if k < 1:
        raise ValueError("RRF k must be positive")

    scores: dict[int, float] = {}
    best_ranks: dict[int, int] = {}
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError("each RRF ranking must contain unique image IDs")
        for rank, image_id in enumerate(ranking, start=1):
            scores[image_id] = scores.get(image_id, 0.0) + 1 / (rank + k)
            best_ranks[image_id] = min(best_ranks.get(image_id, rank), rank)

    return sorted(
        (
            Ranked(image_id=image_id, score=score, best_rank=best_ranks[image_id])
            for image_id, score in scores.items()
        ),
        key=lambda item: (-item.score, item.best_rank, item.image_id),
    )
