from __future__ import annotations

import json
from pathlib import Path

import faiss  # type: ignore[import-untyped]
import numpy as np
import pytest

from mimeme import search
from mimeme.compute.search import Resident
from mimeme.search.model import PreparedLoad


class _Encoder:
    source_model = "test/embed"

    def encode(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def _generation(
    tmp_path: Path,
    version: str,
    *,
    broken: bool = False,
    duplicate_mapping: bool = False,
) -> PreparedLoad:
    root = tmp_path / version
    root.mkdir()
    image_vectors = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    faiss.normalize_L2(image_vectors)
    image = faiss.IndexFlatIP(2)
    image.add(image_vectors)
    faiss.write_index(image, str(root / "index.faiss"))
    mapping = {"0": 1, "1": 1 if duplicate_mapping else 2, "2": 3}
    (root / "mapping.json").write_text(json.dumps(mapping))

    text_vectors = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    faiss.normalize_L2(text_vectors)
    text = faiss.IndexFlatIP(2)
    text.add(text_vectors)
    faiss.write_index(text, str(root / "text_index.faiss"))
    (root / "text_mapping.json").write_text(json.dumps({"0": 2, "1": 3}))

    metadata = {
        "schema_version": 2,
        "version": "wrong" if broken else version,
        "embed_model": "test/embed",
        "dimension": 2,
        "metric": "inner_product",
        "normalized": True,
        "faiss_version": "1.13.2",
        "onnxruntime_version": "1.27.0",
        "encoder_repo": "test/encoder",
        "encoder_revision": "rev-1",
        "encoder_variant": "model.onnx",
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "text_metadata.json").write_text(json.dumps({**metadata, "kind": "text"}))
    return PreparedLoad(
        version=version,
        paths={path.name: str(path) for path in root.iterdir()},
        encoder=search.Encoder(
            repo="test/encoder", revision="rev-1", variant="model.onnx", threads=1
        ),
    )


def _resident() -> Resident:
    return Resident(encoder_factory=lambda _: _Encoder())


def test_load_does_not_serve_until_atomic_switch(tmp_path: Path) -> None:
    resident = _resident()
    loaded = resident.load(_generation(tmp_path, "v1"))

    assert loaded.version == "v1"
    assert resident.status().serving_version is None
    assert resident.status().candidate_version == "v1"

    status = resident.switch("v1")

    assert status.ready is True
    assert status.serving_version == "v1"
    assert status.candidate_version is None
    assert status.encoder_repo == "test/encoder"
    assert status.encoder_revision == "rev-1"
    assert status.encoder_variant == "model.onnx"


def test_failed_load_keeps_the_active_generation_serving(tmp_path: Path) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "v1"))
    resident.switch("v1")

    with pytest.raises(search.Incompatible):
        resident.load(_generation(tmp_path, "v2", broken=True))

    assert resident.status().serving_version == "v1"
    assert resident.status().candidate_version is None


def test_load_rejects_duplicate_stable_image_ids(tmp_path: Path) -> None:
    with pytest.raises(search.Incompatible, match="stable image ID"):
        _resident().load(_generation(tmp_path, "v1", duplicate_mapping=True))


def test_switch_retains_one_generation_for_rollback(tmp_path: Path) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "v1"))
    resident.switch("v1")
    resident.load(_generation(tmp_path, "v2"))
    resident.switch("v2")

    assert resident.status().retained_version == "v1"
    status = resident.rollback("v2")

    assert status.serving_version == "v1"
    assert status.retained_version is None


def test_image_hybrid_and_similar_search_preserve_frozen_order(tmp_path: Path) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "v1"))
    resident.switch("v1")

    image = resident.query(
        search.CandidateRequest(query=search.Query(text="cat", mode="image"), count=10)
    )
    hybrid = resident.query(
        search.CandidateRequest(query=search.Query(text="cat", mode="hybrid"), count=10)
    )
    similar = resident.query(
        search.CandidateRequest(query=search.Query(similar_image_id=1), count=10)
    )

    assert [hit.image_id for hit in image.candidates] == [1, 2, 3]
    assert [hit.image_id for hit in hybrid.candidates] == [2, 3, 1]
    assert [hit.image_id for hit in similar.candidates] == [2, 3]
    assert image.candidates[0].score == pytest.approx(1.0, abs=1e-6)


def test_cursor_returns_the_next_stable_candidate_batch(tmp_path: Path) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "v1"))
    resident.switch("v1")
    query = search.Query(text="cat")

    first = resident.query(search.CandidateRequest(query=query, count=2))
    second = resident.query(search.CandidateRequest(query=query, count=2, cursor=first.cursor))

    assert [hit.image_id for hit in first.candidates] == [1, 2]
    assert [hit.image_id for hit in second.candidates] == [3]
    assert second.exhausted is True


def test_hybrid_first_page_keeps_legacy_requested_k_fusion(tmp_path: Path) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "v1"))
    resident.switch("v1")

    first = resident.query(
        search.CandidateRequest(
            query=search.Query(text="cat", mode="hybrid", limit=1),
            count=1,
        )
    )

    assert [hit.image_id for hit in first.candidates] == [1]
