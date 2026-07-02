from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from text_encoder.metrics import cosine_diagonal
from text_encoder.onnx_encoder import OnnxTextEncoder
from text_encoder.search_sim import SearchSnapshot

K = 10
MODES = ("image", "hybrid")
GATE_MIN_COSINE_QUANT = 0.99
GATE_MIN_COSINE_FP32 = 0.999
GATE_MAX_RECALL_DROP_PTS = 1.0
GATE_MIN_MEDIAN_OVERLAP = 8

ONNX_VARIANTS = [
    ("onnx_int8", "text_model_int8.onnx", GATE_MIN_COSINE_QUANT),
    ("onnx_fp16", "text_model_fp16.onnx", GATE_MIN_COSINE_FP32),
    ("onnx_fp32", "text_model.onnx", GATE_MIN_COSINE_FP32),
]


def recall_at_k(results: list[list[int]], expected: list[int]) -> float:
    hits = sum(1 for hits_row, want in zip(results, expected) if want in hits_row)
    return 100.0 * hits / len(expected)


def median_overlap(results: list[list[int]], reference: list[list[int]]) -> float:
    overlaps = [len(set(row) & set(ref_row)) for row, ref_row in zip(results, reference)]
    return float(np.median(overlaps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-set", type=Path, default=Path("../backend-api/evals/search_eval_set.json")
    )
    parser.add_argument(
        "--snapshot", type=Path, default=Path("data/index-snapshot/v20260227-061522")
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--skip-torch", action="store_true")
    args = parser.parse_args()

    entries = json.loads(args.eval_set.read_text())
    queries = [e["query"] for e in entries]
    expected = [e["expected_image_id"] for e in entries]
    snapshot = SearchSnapshot(args.snapshot)
    print(f"eval set: {len(queries)} queries · snapshot: {snapshot.version} · k={K}")

    from text_encoder.torch_encoder import TorchTextEncoder

    reference_encoder = TorchTextEncoder()
    reference_embeddings = np.stack([reference_encoder.encode(q) for q in queries])
    reference_results = snapshot.exact_search_all(reference_embeddings, K)
    reference_recall = {mode: recall_at_k(reference_results[mode], expected) for mode in MODES}
    reference_hnsw = snapshot.search_all(reference_embeddings, K)
    print(
        f"torch fp32 reference recall@{K} (exact): "
        + "  ".join(f"{mode}={reference_recall[mode]:.1f}%" for mode in MODES)
    )
    print(
        f"torch fp32 reference recall@{K} (prod hnsw, default efSearch): "
        + "  ".join(f"{mode}={recall_at_k(reference_hnsw[mode], expected):.1f}%" for mode in MODES)
    )
    print(
        "gates are measured on exact search: HNSW default-efSearch retrieval is unstable under "
        "tiny query perturbations and its variant deltas are traversal noise (see eval-report)"
    )

    rows = []
    for name, filename, min_cosine_gate in ONNX_VARIANTS:
        model_path = args.artifacts / filename
        if not model_path.exists():
            print(f"skipping {name}: {model_path} missing")
            continue
        encoder = OnnxTextEncoder(model_path)
        embeddings = np.stack([encoder.encode(q) for q in queries])
        cosines = cosine_diagonal(reference_embeddings, embeddings)
        results = snapshot.exact_search_all(embeddings, K)

        row = {
            "variant": name,
            "size_mb": model_path.stat().st_size / 2**20,
            "min_cosine": float(cosines.min()),
            "mean_cosine": float(cosines.mean()),
            "gates": [],
        }
        if cosines.min() < min_cosine_gate:
            row["gates"].append(f"cosine {cosines.min():.4f} < {min_cosine_gate}")
        for mode in MODES:
            recall = recall_at_k(results[mode], expected)
            drop = reference_recall[mode] - recall
            overlap = median_overlap(results[mode], reference_results[mode])
            row[f"recall_{mode}"] = recall
            row[f"drop_{mode}"] = drop
            row[f"overlap_{mode}"] = overlap
            if drop > GATE_MAX_RECALL_DROP_PTS:
                row["gates"].append(f"{mode} recall drop {drop:.1f} > {GATE_MAX_RECALL_DROP_PTS}")
            if overlap < GATE_MIN_MEDIAN_OVERLAP:
                row["gates"].append(f"{mode} overlap {overlap} < {GATE_MIN_MEDIAN_OVERLAP}")
        rows.append(row)

    print()
    header = (
        f"| variant | size MB | min cos | mean cos | "
        f"recall@{K} img | Δ img | recall@{K} hyb | Δ hyb | ovl img | ovl hyb | gates |"
    )
    print(header)
    print("|" + "---|" * 11)
    print(
        f"| torch_fp32 (ref) | — | 1.0000 | 1.0000 | {reference_recall['image']:.1f}% | — | "
        f"{reference_recall['hybrid']:.1f}% | — | {K} | {K} | reference |"
    )
    for row in rows:
        status = "PASS" if not row["gates"] else "FAIL: " + "; ".join(row["gates"])
        print(
            f"| {row['variant']} | {row['size_mb']:.0f} | {row['min_cosine']:.4f} | "
            f"{row['mean_cosine']:.4f} | {row['recall_image']:.1f}% | {row['drop_image']:+.1f} | "
            f"{row['recall_hybrid']:.1f}% | {row['drop_hybrid']:+.1f} | "
            f"{row['overlap_image']:.0f}/{K} | {row['overlap_hybrid']:.0f}/{K} | {status} |"
        )

    passing = [row for row in rows if not row["gates"]]
    if passing:
        selected = passing[0]
        print(
            f"\nselected variant (first passing in ladder order): "
            f"{selected['variant']} ({selected['size_mb']:.0f} MB)"
        )
    else:
        print("\nno ONNX variant passed all gates")


if __name__ == "__main__":
    main()
