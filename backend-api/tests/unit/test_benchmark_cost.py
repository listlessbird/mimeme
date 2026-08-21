from __future__ import annotations

import pytest
from scripts.benchmark_cost import estimate


def test_estimate_separates_processing_cold_start_and_always_on_costs() -> None:
    report = estimate(
        throughput=2,
        gpu_price_per_hour=0.20,
        monthly_images=[1_000, 10_000],
        cold_start_seconds=20,
        runs_per_month=30,
    )

    assert report["processing_cost_per_1000"] == pytest.approx(0.027778)
    assert report["cold_start_cost_per_run"] == pytest.approx(0.001111)
    assert report["always_on_gpu_cost"] == pytest.approx(146)
    assert report["monthly"] == [
        {
            "images": 1_000,
            "processing_gpu_hours": 0.138889,
            "serverless_gpu_hours": 0.305556,
            "serverless_gpu_cost": 0.061111,
        },
        {
            "images": 10_000,
            "processing_gpu_hours": 1.388889,
            "serverless_gpu_hours": 1.555556,
            "serverless_gpu_cost": 0.311111,
        },
    ]


def test_estimate_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="throughput"):
        estimate(throughput=0, gpu_price_per_hour=0.2, monthly_images=[])
