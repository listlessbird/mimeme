#!/usr/bin/env python3
"""Submit real image URLs to Temporal ingestion and report end-to-end throughput.

Run this against an isolated/local API stack. Per-stage p50/p95 values are read
from JSON worker logs supplied with --worker-log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import httpx


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--worker-log", type=Path)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0)
    parser.add_argument("--gpu-price-per-hour", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return (
        values[low]
        if low == high
        else values[low] + (values[high] - values[low]) * (position - low)
    )


async def _sample_gpu(
    samples: list[dict[str, float]], stop: asyncio.Event, interval: float
) -> None:
    while not stop.is_set():
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0 and stdout:
            utilization, used, total = (
                float(value.strip()) for value in stdout.decode().splitlines()[0].split(",")
            )
            samples.append(
                {
                    "gpu_utilization_percent": utilization,
                    "vram_used_mb": used,
                    "vram_total_mb": total,
                }
            )
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass


def _stage_metrics(path: Path | None, job_ids: set[str]) -> dict[str, object]:
    values = {"annotation_ms": [], "embedding_ms": [], "total_ms": []}
    if path and path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "ingest_item_completed" or event.get("job_id") not in job_ids:
                continue
            for key in values:
                value = event.get(key)
                if isinstance(value, int | float):
                    values[key].append(float(value))
    return {
        key: {
            "count": len(series),
            "p50_ms": round(_percentile(series, 0.5) or 0, 2) if series else None,
            "p95_ms": round(_percentile(series, 0.95) or 0, 2) if series else None,
        }
        for key, series in values.items()
    }


async def _main() -> None:
    args = _args()
    entries = json.loads(args.manifest.read_text())[: args.limit]
    if not entries:
        raise SystemExit("manifest contains no images")
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    job_ids: list[str] = []
    gpu_samples: list[dict[str, float]] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(_sample_gpu(gpu_samples, stop, args.gpu_sample_interval))
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(base_url=args.api_url, headers=headers, timeout=60) as http:
            for offset in range(0, len(entries), 100):
                response = await http.post(
                    "/images",
                    json={
                        "urls": [entry["public_url"] for entry in entries[offset : offset + 100]],
                        "dataset": f"benchmark-temporal-{int(started)}",
                        "tags": ["benchmark"],
                    },
                )
                response.raise_for_status()
                job_ids.append(response.json()["job_id"])
            terminal = {"COMPLETED", "FAILED", "CANCELED"}
            states: dict[str, dict] = {}
            while True:
                for job_id in job_ids:
                    response = await http.get(f"/jobs/{job_id}")
                    response.raise_for_status()
                    states[job_id] = response.json()
                if all(state["status"] in terminal for state in states.values()):
                    break
                await asyncio.sleep(args.poll_interval)
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        await sampler

    images_per_second = len(entries) / elapsed
    summary: dict[str, object] = {
        "images": len(entries),
        "job_ids": job_ids,
        "elapsed_seconds": round(elapsed, 2),
        "images_per_minute": round(images_per_second * 60, 2),
        "images_per_second": round(images_per_second, 4),
        "stages": _stage_metrics(args.worker_log, set(job_ids)),
        "gpu_utilization_mean_percent": round(
            statistics.fmean(sample["gpu_utilization_percent"] for sample in gpu_samples), 2
        )
        if gpu_samples
        else None,
        "gpu_utilization_peak_percent": max(
            (sample["gpu_utilization_percent"] for sample in gpu_samples), default=None
        ),
        "vram_peak_mb": max((sample["vram_used_mb"] for sample in gpu_samples), default=None),
    }
    if args.gpu_price_per_hour is not None:
        gpu_hours = (1000 / images_per_second) / 3600
        summary["cost_per_1000"] = round(gpu_hours * args.gpu_price_per_hour, 6)
        summary["gpu_hours_per_1000"] = round(gpu_hours, 6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**summary, "gpu_samples": gpu_samples}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
