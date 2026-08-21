#!/usr/bin/env python3
"""Measure client-observed inference.Local latency through mimeme-compute."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
import uuid
from pathlib import Path

import httpx

from mimeme.inference.local import Local
from mimeme.inference.model import Batch, Input, Item


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--embed-model", default="google/siglip2-base-patch16-naflex")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--gateway-log", type=Path)
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


async def _main() -> None:
    args = _args()
    entries = json.loads(args.manifest.read_text())[: args.limit]
    run_id = uuid.uuid4().hex[:12]
    rows: list[dict[str, object]] = []
    timeout = httpx.Timeout(900)
    async with httpx.AsyncClient(timeout=timeout) as http:
        client = Local(
            http,
            base_url=args.base_url,
            embed_model=args.embed_model,
            poll_interval_s=args.poll_interval,
        )
        if not await client.ready():
            raise SystemExit(f"inference gateway is not ready: {args.base_url}")

        captions: dict[int, str] = {}
        for index, entry in enumerate(entries):
            started = time.perf_counter()
            annotation = await client.annotate(Input(image_id=index, media_key=entry["media_key"]))
            elapsed = (time.perf_counter() - started) * 1000
            captions[index] = " ".join(
                part for part in (annotation.caption, annotation.ocr_text) if part
            )
            rows.append(
                {
                    "operation": "annotation",
                    "media_key": entry["media_key"],
                    "images": 1,
                    "elapsed_ms": round(elapsed, 2),
                    "elapsed_ms_per_image": round(elapsed, 2),
                }
            )

        for offset in range(0, len(entries), args.batch_size):
            chunk = entries[offset : offset + args.batch_size]
            batch = Batch(
                dataset=f"benchmark-{run_id}",
                items=[
                    Item(
                        image_id=offset + index,
                        media_key=entry["media_key"],
                        sha256=entry["sha256"],
                        text=captions[offset + index],
                        dataset=f"benchmark-{run_id}",
                    )
                    for index, entry in enumerate(chunk)
                ],
            )
            started = time.perf_counter()
            result = await client.embed(batch)
            elapsed = (time.perf_counter() - started) * 1000
            failures = [item for item in result.items if getattr(item, "error", None)]
            if failures:
                raise RuntimeError(str(failures[0]))
            rows.append(
                {
                    "operation": "embedding",
                    "images": len(chunk),
                    "elapsed_ms": round(elapsed, 2),
                    "elapsed_ms_per_image": round(elapsed / len(chunk), 2),
                }
            )

    summary: dict[str, object] = {
        "run_id": run_id,
        "base_url": args.base_url,
        "batch_size": args.batch_size,
    }
    for operation in ("annotation", "embedding"):
        values = [
            float(row["elapsed_ms_per_image"]) for row in rows if row["operation"] == operation
        ]
        summary[operation] = {
            "images": sum(int(row["images"]) for row in rows if row["operation"] == operation),
            "mean_ms": round(statistics.fmean(values), 2),
            "p50_ms": round(_percentile(values, 0.5) or 0, 2),
            "p95_ms": round(_percentile(values, 0.95) or 0, 2),
            "images_per_second": round(1000 / statistics.fmean(values), 3),
        }
    gateway_events: list[dict[str, object]] = []
    if args.gateway_log and args.gateway_log.exists():
        for line in args.gateway_log.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "compute_inference_job_completed":
                continue
            gateway_events.append(event)
        component_keys = (
            "media_download_ms",
            "gpu_queue_wait_ms",
            "gpu_model_load_ms",
            "image_decode_ms",
            "vision_encode_ms",
            "caption_ms",
            "ocr_ms",
            "siglip_preprocess_ms",
            "siglip_image_ms",
            "siglip_text_ms",
            "artifact_upload_ms",
            "duration_ms",
        )
        summary["gateway_components"] = {}
        for operation in ("annotate", "embed"):
            operation_events = [
                event for event in gateway_events if event.get("inference_operation") == operation
            ]
            summary["gateway_components"][operation] = {
                key: {
                    "p50_ms": round(_percentile(values, 0.5) or 0, 2),
                    "p95_ms": round(_percentile(values, 0.95) or 0, 2),
                }
                for key in component_keys
                if (
                    values := [
                        float(event[key])
                        for event in operation_events
                        if isinstance(event.get(key), int | float)
                    ]
                )
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({**summary, "rows": rows, "gateway_events": gateway_events}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
