#!/usr/bin/env python3
"""Benchmark Moondream and SigLIP directly, with no gateway or storage I/O."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

import torch

from mimeme.compute.inference import Models
from mimeme.compute.model import AnnotateCall, EmbedCall, EmbedCallItem
from mimeme.config import InferenceConfig

CONFIGS = {
    "A": ("swap", 1),
    "B": ("both", 1),
    "C": ("both", 2),
    "D": ("both", 4),
    "E": ("both", 8),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotation-limit", type=int, default=0, help="0 uses --limit")
    parser.add_argument("--skip-annotations", action="store_true")
    return parser.parse_args()


def _sync() -> None:
    torch.cuda.synchronize()


def _start() -> float:
    torch.cuda.reset_peak_memory_stats()
    _sync()
    return time.perf_counter()


def _finish(started: float) -> tuple[float, float, float]:
    _sync()
    divisor = 1024 * 1024
    return (
        (time.perf_counter() - started) * 1000,
        torch.cuda.max_memory_allocated() / divisor,
        torch.cuda.max_memory_reserved() / divisor,
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _summary(rows: list[dict[str, object]], config: str, residency: str, batch: int) -> dict:
    result: dict[str, object] = {
        "config": config,
        "residency": residency,
        "siglip_batch_size": batch,
        "gpu": torch.cuda.get_device_name(),
        "status": "ok",
    }
    for operation in ("annotation", "embedding"):
        selected = [row for row in rows if row["operation"] == operation]
        timings = [float(row["elapsed_ms_per_image"]) for row in selected]
        result[operation] = {
            "images": sum(int(row["images"]) for row in selected),
            "mean_ms": round(statistics.fmean(timings), 2) if timings else None,
            "p50_ms": round(_percentile(timings, 0.50) or 0, 2) if timings else None,
            "p95_ms": round(_percentile(timings, 0.95) or 0, 2) if timings else None,
            "images_per_second": round(1000 / statistics.fmean(timings), 3) if timings else None,
            "peak_allocated_mb": round(
                max((float(row["peak_allocated_mb"]) for row in selected), default=0), 2
            ),
            "peak_reserved_mb": round(
                max((float(row["peak_reserved_mb"]) for row in selected), default=0), 2
            ),
        }
    result["model_loads"] = [row for row in rows if row["operation"] == "model_load"]
    result["both_models_below_8gb"] = (
        residency == "both"
        and max((float(row["peak_reserved_mb"]) for row in rows), default=0) < 8192
    )
    return result


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    entries = json.loads(args.manifest.read_text())[: args.limit]
    if not entries:
        raise SystemExit("manifest contains no images")
    residency, batch = CONFIGS[args.config]
    config = InferenceConfig(residency=residency, embed_batch_size=batch)
    models = Models(config)
    rows: list[dict[str, object]] = []

    try:
        started = _start()
        models._load_vision()
        elapsed, allocated, reserved = _finish(started)
        rows.append(
            {
                "operation": "model_load",
                "model": "moondream",
                "elapsed_ms": round(elapsed, 2),
                "peak_allocated_mb": round(allocated, 2),
                "peak_reserved_mb": round(reserved, 2),
            }
        )

        if residency == "both":
            started = _start()
            models._load_embed()
            elapsed, allocated, reserved = _finish(started)
            rows.append(
                {
                    "operation": "model_load",
                    "model": "siglip",
                    "elapsed_ms": round(elapsed, 2),
                    "peak_allocated_mb": round(allocated, 2),
                    "peak_reserved_mb": round(reserved, 2),
                }
            )

        captions: list[str] = []
        annotation_entries = (
            [] if args.skip_annotations else entries[: args.annotation_limit or len(entries)]
        )
        for entry in annotation_entries:
            started = _start()
            reply = models.annotate(AnnotateCall(path=entry["path"]))
            elapsed, allocated, reserved = _finish(started)
            captions.append(" ".join(part for part in (reply.caption, reply.ocr_text) if part))
            rows.append(
                {
                    "operation": "annotation",
                    "image": entry["path"],
                    "images": 1,
                    "elapsed_ms": round(elapsed, 2),
                    "elapsed_ms_per_image": round(elapsed, 2),
                    "peak_allocated_mb": round(allocated, 2),
                    "peak_reserved_mb": round(reserved, 2),
                    "telemetry": reply.telemetry.model_dump() if reply.telemetry else None,
                    "caption": reply.caption,
                    "ocr_text": reply.ocr_text,
                }
            )

        if residency == "swap":
            started = _start()
            models._load_embed()
            elapsed, allocated, reserved = _finish(started)
            rows.append(
                {
                    "operation": "model_load",
                    "model": "siglip",
                    "elapsed_ms": round(elapsed, 2),
                    "peak_allocated_mb": round(allocated, 2),
                    "peak_reserved_mb": round(reserved, 2),
                }
            )

        with tempfile.TemporaryDirectory(prefix="mimeme-model-bench-") as tmp:
            output_dir = Path(tmp)
            for offset in range(0, len(entries), batch):
                chunk = entries[offset : offset + batch]
                items = [
                    EmbedCallItem(
                        image_id=offset + index,
                        path=entry["path"],
                        text=(
                            captions[offset + index]
                            if offset + index < len(captions)
                            else "meme image"
                        ),
                        image_out=str(output_dir / f"{offset + index}-image.npy"),
                        text_out=str(output_dir / f"{offset + index}-text.npy"),
                    )
                    for index, entry in enumerate(chunk)
                ]
                started = _start()
                reply = models.embed(EmbedCall(items=items))
                elapsed, allocated, reserved = _finish(started)
                failures = [item.error for item in reply.items if not item.ok]
                if failures:
                    raise RuntimeError(f"embedding failed: {failures[0]}")
                rows.append(
                    {
                        "operation": "embedding",
                        "images": len(chunk),
                        "elapsed_ms": round(elapsed, 2),
                        "elapsed_ms_per_image": round(elapsed / len(chunk), 2),
                        "peak_allocated_mb": round(allocated, 2),
                        "peak_reserved_mb": round(reserved, 2),
                        "telemetry": reply.telemetry.model_dump() if reply.telemetry else None,
                    }
                )
    except torch.OutOfMemoryError as exc:
        result = {
            "config": args.config,
            "residency": residency,
            "siglip_batch_size": batch,
            "gpu": torch.cuda.get_device_name(),
            "status": "oom",
            "error": str(exc),
            "rows": rows,
        }
    except Exception as exc:
        result = {
            "config": args.config,
            "residency": residency,
            "siglip_batch_size": batch,
            "gpu": torch.cuda.get_device_name(),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "rows": rows,
        }
    else:
        result = {
            **_summary(rows, args.config, residency, batch),
            "vision_compile": config.vision_compile,
            "rows": rows,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
