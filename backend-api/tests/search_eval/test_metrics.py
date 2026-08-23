import pytest

from mimeme.search_eval.metrics import calculate


def test_calculates_graded_ranking_metrics() -> None:
    metrics = calculate(
        rankings={1: [10, 11, 12], 2: [20, 21]},
        judgments={1: {10: 3, 11: 1, 12: 2}, 2: {20: 0, 21: 2}},
        latencies_ms={1: 10, 2: 30},
    )

    assert metrics.query_count == 2
    assert metrics.ndcg_at_10 == pytest.approx(0.802, abs=0.001)
    assert metrics.precision_at_5 == pytest.approx(0.3)
    assert metrics.success_at_5 == 1
    assert metrics.mrr_at_10 == 0.75
    assert metrics.judged_at_10 == 1
    assert metrics.latency_p50_ms == 20
    assert metrics.latency_p95_ms == 29


def test_rejects_query_without_relevant_judgment() -> None:
    with pytest.raises(ValueError, match="no relevant judgments"):
        calculate(
            rankings={1: [10]},
            judgments={1: {10: 1}},
            latencies_ms={1: 10},
        )


def test_grade_zero_is_judged_but_an_absent_qrel_is_not() -> None:
    metrics = calculate(
        rankings={1: [10, 11, 12]},
        judgments={1: {10: 0, 12: 2}},
        latencies_ms={1: 10},
    )

    assert metrics.judged_at_10 == pytest.approx(2 / 3)


def test_reciprocal_rank_stops_at_ten() -> None:
    metrics = calculate(
        rankings={1: [*range(1, 11), 99]},
        judgments={1: {99: 3}},
        latencies_ms={1: 10},
    )

    assert metrics.mrr_at_10 == 0


def test_requires_a_latency_for_every_query() -> None:
    with pytest.raises(ValueError, match="every query requires"):
        calculate(
            rankings={1: [10]},
            judgments={1: {10: 3}},
            latencies_ms={},
        )
