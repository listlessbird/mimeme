from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from text_encoder.queries import BENCH_QUERIES
from text_encoder.torch_encoder import TorchTextEncoder

DEFAULT_OUT = (
    Path(__file__).parent.parent.parent
    / "backend-api"
    / "model_smoke"
    / "fixtures"
    / "torch_text_reference.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    encoder = TorchTextEncoder()
    input_ids = np.stack(
        [encoder.tokenize(q)["input_ids"].numpy()[0] for q in BENCH_QUERIES]
    ).astype(np.int64)
    embeddings = np.stack([encoder.encode(q) for q in BENCH_QUERIES])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        queries=np.array(BENCH_QUERIES),
        input_ids=input_ids,
        embeddings=embeddings,
    )
    print(f"wrote {args.out} ({len(BENCH_QUERIES)} queries)")


if __name__ == "__main__":
    main()
