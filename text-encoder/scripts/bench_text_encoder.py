from __future__ import annotations

import argparse
import time

import numpy as np

from text_encoder.queries import BENCH_QUERIES, FINGERPRINT_QUERY
from text_encoder.torch_encoder import MODEL_ID, TorchTextEncoder


def process_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    load_started = time.monotonic()
    encoder = TorchTextEncoder(model_id=args.model_id, device=args.device)
    load_ms = (time.monotonic() - load_started) * 1000

    inputs = encoder.tokenize(FINGERPRINT_QUERY)
    input_signature = {k: (tuple(v.shape), str(v.dtype)) for k, v in inputs.items()}

    for _ in range(args.warmup):
        encoder.encode(FINGERPRINT_QUERY)

    timings: dict[str, list[float]] = {q: [] for q in BENCH_QUERIES}
    for _ in range(args.rounds):
        for query in BENCH_QUERIES:
            started = time.monotonic()
            encoder.encode(query)
            timings[query].append((time.monotonic() - started) * 1000)

    fingerprint = encoder.encode(FINGERPRINT_QUERY)[:8]
    rss_mb = process_rss_mb()
    all_samples = np.array([ms for samples in timings.values() for ms in samples])

    print(f"model: {args.model_id}  device: {args.device}")
    print(f"load_ms: {load_ms:.0f}")
    print(f"encoder inputs: {input_signature}")
    print()
    print(f"{'query':<60} {'p50 ms':>8} {'p95 ms':>8}")
    for query, samples in timings.items():
        arr = np.array(samples)
        label = query if len(query) <= 57 else query[:57] + "..."
        print(f"{label:<60} {np.percentile(arr, 50):>8.1f} {np.percentile(arr, 95):>8.1f}")
    print()
    print(
        f"overall ({len(all_samples)} samples, {args.rounds} rounds x {len(BENCH_QUERIES)} queries):"
        f" p50={np.percentile(all_samples, 50):.1f}ms p95={np.percentile(all_samples, 95):.1f}ms"
    )
    print(f"rss_mb: {rss_mb:.0f}")
    np.set_printoptions(precision=6, suppress=False)
    print(f"fingerprint '{FINGERPRINT_QUERY}' first 8 dims: {fingerprint}")


if __name__ == "__main__":
    main()
