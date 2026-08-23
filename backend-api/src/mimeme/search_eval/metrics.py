from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import ir_measures
from ir_measures import RR, Judged, P, Success, nDCG
from pydantic import BaseModel, ConfigDict, Field

# Provider details stay in this adapter. Callers only use Mimeme's named metrics.
_NDCG_10 = nDCG(gains={0: 0, 1: 1, 2: 3, 3: 7}) @ 10
_PRECISION_5 = P(rel=2) @ 5
_SUCCESS_5 = Success(rel=2) @ 5
_RR_10 = RR(rel=2) @ 10
_JUDGED_10 = Judged @ 10
_MEASURES = [_NDCG_10, _PRECISION_5, _SUCCESS_5, _RR_10, _JUDGED_10]


class QueryMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: int
    ndcg_at_10: float
    precision_at_5: float
    success_at_5: bool
    reciprocal_rank_at_10: float
    judged_at_10: float


class Metrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_count: int = Field(ge=1)
    ndcg_at_10: float
    precision_at_5: float
    success_at_5: float
    mrr_at_10: float
    judged_at_10: float
    latency_p50_ms: float
    latency_p95_ms: float
    per_query: list[QueryMetrics]


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def calculate(
    rankings: Mapping[int, Sequence[int]],
    judgments: Mapping[int, Mapping[int, int]],
    latencies_ms: Mapping[int, float],
) -> Metrics:
    query_ids = sorted(judgments)
    if not query_ids:
        raise ValueError("metrics require at least one query")
    if set(latencies_ms) != set(query_ids):
        raise ValueError("every query requires one latency measurement")
    for query_id in query_ids:
        if not any(grade >= 2 for grade in judgments[query_id].values()):
            raise ValueError(f"query {query_id} has no relevant judgments")

    qrels = [
        ir_measures.Qrel(f"q:{query_id}", f"img:{image_id}", grade)
        for query_id in query_ids
        for image_id, grade in judgments[query_id].items()
    ]
    run = [
        ir_measures.ScoredDoc(
            f"q:{query_id}",
            f"img:{image_id}",
            float(10_000 - rank),
        )
        for query_id in query_ids
        for rank, image_id in enumerate(rankings.get(query_id, ()), 1)
    ]

    aggregate = ir_measures.calc_aggregate(_MEASURES, qrels, run)
    values: dict[int, dict[object, float]] = {query_id: {} for query_id in query_ids}
    for result in ir_measures.iter_calc(_MEASURES, qrels, run):
        values[int(result.query_id.removeprefix("q:"))][result.measure] = result.value

    per_query = [
        QueryMetrics(
            query_id=query_id,
            ndcg_at_10=values[query_id][_NDCG_10],
            precision_at_5=values[query_id][_PRECISION_5],
            success_at_5=bool(values[query_id][_SUCCESS_5]),
            reciprocal_rank_at_10=values[query_id][_RR_10],
            judged_at_10=values[query_id][_JUDGED_10],
        )
        for query_id in query_ids
    ]
    latencies = list(latencies_ms.values())
    return Metrics(
        query_count=len(query_ids),
        ndcg_at_10=aggregate[_NDCG_10],
        precision_at_5=aggregate[_PRECISION_5],
        success_at_5=aggregate[_SUCCESS_5],
        mrr_at_10=aggregate[_RR_10],
        judged_at_10=aggregate[_JUDGED_10],
        latency_p50_ms=_percentile(latencies, 0.5),
        latency_p95_ms=_percentile(latencies, 0.95),
        per_query=per_query,
    )
