from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_encoder.onnx_encoder import OnnxTextEncoder
from text_encoder.queries import BENCH_QUERIES, FINGERPRINT_QUERY


def process_rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    load_started = time.monotonic()
    encoder = OnnxTextEncoder(args.model, intra_op_threads=args.threads)
    load_ms = (time.monotonic() - load_started) * 1000

    for _ in range(args.warmup):
        encoder.encode(FINGERPRINT_QUERY)

    samples = []
    for _ in range(args.rounds):
        for query in BENCH_QUERIES:
            started = time.monotonic()
            encoder.encode(query)
            samples.append((time.monotonic() - started) * 1000)

    fingerprint = encoder.encode(FINGERPRINT_QUERY)[:8]
    arr = np.array(samples)
    print(f"model: {args.model.name}  threads: {args.threads or 'default'}")
    print(f"load_ms: {load_ms:.0f}")
    print(
        f"overall ({len(arr)} samples): p50={np.percentile(arr, 50):.1f}ms "
        f"p95={np.percentile(arr, 95):.1f}ms"
    )
    print(f"rss_mb: {process_rss_mb():.0f}")
    np.set_printoptions(precision=6, suppress=False)
    print(f"fingerprint '{FINGERPRINT_QUERY}' first 8 dims: {fingerprint}")


if __name__ == "__main__":
    main()
