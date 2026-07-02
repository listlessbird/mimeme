from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from text_encoder.metrics import (
    cosine_diagonal,
    neighbor_overlap,
    pairwise_cosine,
    top_k_neighbors,
)
from text_encoder.onnx_encoder import OnnxTextEncoder
from text_encoder.queries import BENCH_QUERIES
from text_encoder.torch_encoder import EMBED_DIM, TorchTextEncoder

pytestmark = [
    pytest.mark.model_smoke,
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_SMOKE") != "1",
        reason="set RUN_MODEL_SMOKE=1 or run `just test-model`",
    ),
]


class Encoder(Protocol):
    def encode(self, query: str) -> np.ndarray: ...


class Candidate:
    def __init__(self, factory: Callable[[], Encoder], min_cosine: float) -> None:
        self.factory = factory
        self.min_cosine = min_cosine


ARTIFACTS = Path(__file__).parent.parent / "artifacts"

CANDIDATES: dict[str, Candidate] = {
    name: Candidate(lambda p=path: OnnxTextEncoder(p), min_cosine)
    for name, filename, min_cosine in [
        ("onnx_fp32", "text_model.onnx", 0.999),
        ("onnx_fp16", "text_model_fp16.onnx", 0.999),
        ("onnx_int8", "text_model_int8.onnx", 0.99),
    ]
    if (path := ARTIFACTS / filename).exists()
}

TOP_K = 3
MIN_NEIGHBOR_OVERLAP = 2


@pytest.fixture(scope="module")
def reference() -> TorchTextEncoder:
    return TorchTextEncoder()


@pytest.fixture(scope="module")
def reference_embeddings(reference: TorchTextEncoder) -> np.ndarray:
    return np.stack([reference.encode(q) for q in BENCH_QUERIES])


def test_reference_embedding_contract(reference_embeddings: np.ndarray) -> None:
    assert reference_embeddings.shape == (len(BENCH_QUERIES), EMBED_DIM)
    assert reference_embeddings.dtype == np.float32
    assert np.isfinite(reference_embeddings).all()
    norms = np.linalg.norm(reference_embeddings, axis=1)
    assert (norms > 0).all()


def test_reference_queries_are_distinct(reference_embeddings: np.ndarray) -> None:
    similarity = pairwise_cosine(reference_embeddings)
    off_diagonal = similarity[~np.eye(len(BENCH_QUERIES), dtype=bool)]
    assert off_diagonal.max() < 0.999


def test_tokenizer_parity_with_processor(reference: TorchTextEncoder) -> None:
    if not (ARTIFACTS / "tokenizer.json").exists():
        pytest.skip("no exported tokenizer.json (run scripts/export_text_onnx.py)")
    model_path = next(
        (
            p
            for p in (ARTIFACTS / "text_model_int8.onnx", ARTIFACTS / "text_model.onnx")
            if p.exists()
        ),
        None,
    )
    if model_path is None:
        pytest.skip("no exported ONNX model")
    encoder = OnnxTextEncoder(model_path)
    for query in BENCH_QUERIES:
        torch_ids = reference.tokenize(query)["input_ids"].numpy()[0]
        onnx_ids = encoder.tokenize(query)[0]
        np.testing.assert_array_equal(onnx_ids, torch_ids, err_msg=f"query: {query!r}")


@pytest.mark.parametrize(
    "name",
    sorted(CANDIDATES)
    or [
        pytest.param(
            "none",
            marks=pytest.mark.skip(reason="no candidate encoders registered yet (issues 003/005)"),
        )
    ],
)
def test_candidate_matches_reference(name: str, reference_embeddings: np.ndarray) -> None:
    candidate = CANDIDATES[name]
    encoder = candidate.factory()
    embeddings = np.stack([encoder.encode(q) for q in BENCH_QUERIES])

    assert embeddings.shape == reference_embeddings.shape
    assert np.isfinite(embeddings).all()

    cosines = cosine_diagonal(reference_embeddings, embeddings)
    assert cosines.min() >= candidate.min_cosine, (
        f"{name}: min cosine {cosines.min():.5f} < {candidate.min_cosine}"
    )

    reference_neighbors = top_k_neighbors(pairwise_cosine(reference_embeddings), TOP_K)
    candidate_neighbors = top_k_neighbors(pairwise_cosine(embeddings), TOP_K)
    overlaps = neighbor_overlap(reference_neighbors, candidate_neighbors)
    assert (overlaps >= MIN_NEIGHBOR_OVERLAP).all(), (
        f"{name}: top-{TOP_K} neighbor overlap below {MIN_NEIGHBOR_OVERLAP}/{TOP_K} "
        f"for queries {[BENCH_QUERIES[i] for i in np.where(overlaps < MIN_NEIGHBOR_OVERLAP)[0]]}"
    )
