#!/usr/bin/env python3
"""Turn measured ingestion throughput into serverless and always-on GPU costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MONTHLY_VOLUMES = (1_000, 10_000, 100_000, 1_000_000)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--result", type=Path, help="Temporal benchmark JSON")
    source.add_argument("--throughput", type=float, help="Measured images per second")
    parser.add_argument("--gpu-price-per-hour", type=float, required=True)
    parser.add_argument(
        "--monthly-images",
        type=int,
        nargs="+",
        default=DEFAULT_MONTHLY_VOLUMES,
    )
    parser.add_argument(
        "--cold-start-seconds",
        type=float,
        default=0,
        help="Warm-benchmark adjustment applied once per ingestion run",
    )
    parser.add_argument("--runs-per-month", type=int, default=1)
    parser.add_argument("--always-on-hours", type=float, default=730)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _throughput(args: argparse.Namespace) -> float:
    if args.throughput is not None:
        value = args.throughput
    else:
        result = json.loads(args.result.read_text())
        value = result.get("images_per_second")
    if not isinstance(value, int | float) or value <= 0:
        raise SystemExit("throughput must be a positive number")
    return float(value)


def estimate(
    *,
    throughput: float,
    gpu_price_per_hour: float,
    monthly_images: list[int] | tuple[int, ...],
    cold_start_seconds: float = 0,
    runs_per_month: int = 1,
    always_on_hours: float = 730,
) -> dict[str, object]:
    if throughput <= 0:
        raise ValueError("throughput must be positive")
    if gpu_price_per_hour < 0 or cold_start_seconds < 0 or always_on_hours < 0:
        raise ValueError("prices and durations cannot be negative")
    if runs_per_month < 0 or any(images < 0 for images in monthly_images):
        raise ValueError("run and image counts cannot be negative")

    processing_seconds_per_1000 = 1000 / throughput
    processing_cost_per_1000 = processing_seconds_per_1000 / 3600 * gpu_price_per_hour
    cold_start_cost_per_run = cold_start_seconds / 3600 * gpu_price_per_hour
    monthly: list[dict[str, int | float]] = []
    for images in monthly_images:
        processing_seconds = images / throughput
        serverless_seconds = processing_seconds + cold_start_seconds * runs_per_month
        monthly.append(
            {
                "images": images,
                "processing_gpu_hours": round(processing_seconds / 3600, 6),
                "serverless_gpu_hours": round(serverless_seconds / 3600, 6),
                "serverless_gpu_cost": round(serverless_seconds / 3600 * gpu_price_per_hour, 6),
            }
        )

    always_on_cost = always_on_hours * gpu_price_per_hour
    available_processing_seconds = max(
        0.0, always_on_hours * 3600 - cold_start_seconds * runs_per_month
    )
    return {
        "throughput_images_per_second": throughput,
        "gpu_price_per_hour": gpu_price_per_hour,
        "processing_seconds_per_1000": round(processing_seconds_per_1000, 3),
        "processing_gpu_hours_per_1000": round(processing_seconds_per_1000 / 3600, 6),
        "processing_cost_per_1000": round(processing_cost_per_1000, 6),
        "cold_start_seconds": cold_start_seconds,
        "cold_start_cost_per_run": round(cold_start_cost_per_run, 6),
        "runs_per_month": runs_per_month,
        "monthly": monthly,
        "always_on_hours": always_on_hours,
        "always_on_gpu_cost": round(always_on_cost, 6),
        "break_even_images_per_month": round(available_processing_seconds * throughput),
    }


def main() -> None:
    args = _args()
    report = estimate(
        throughput=_throughput(args),
        gpu_price_per_hour=args.gpu_price_per_hour,
        monthly_images=args.monthly_images,
        cold_start_seconds=args.cold_start_seconds,
        runs_per_month=args.runs_per_month,
        always_on_hours=args.always_on_hours,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
