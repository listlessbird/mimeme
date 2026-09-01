from __future__ import annotations

import json
from pathlib import Path

import faiss  # type: ignore[import-untyped]
import numpy as np
import pytest

from mimeme import search
from mimeme.compute.search import Resident
from mimeme.index import bm25
from mimeme.inference import bge
from mimeme.search import generation_workspace
from mimeme.search.document import SearchDocument
from mimeme.search.model import PreparedLoad


class _Encoder:
    source_model = "test/embed"

    def encode(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class _BgeEncoder:
    source_model = bge.SOURCE_MODEL

    def encode(self, text: str) -> np.ndarray:
        return np.array([1.0, *([0.0] * (bge.DIMENSION - 1))], dtype=np.float32)


def _generation(
    tmp_path: Path,
    version: str,
    *,
    broken: bool = False,
    duplicate_mapping: bool = False,
    revision: str = "rev-1",
    corrupt_bm25: bool = False,
    with_bge: bool = False,
) -> PreparedLoad:
    root = generation_workspace.prepare(tmp_path, version)
    image_vectors = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    faiss.normalize_L2(image_vectors)
    image = faiss.IndexFlatIP(2)
    image.add(image_vectors)
    faiss.write_index(image, str(root / "index.faiss"))
    mapping = {"0": 1, "1": 1 if duplicate_mapping else 2, "2": 3}
    (root / "mapping.json").write_text(json.dumps(mapping))

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
        "encoder_revision": revision,
        "encoder_variant": "model.onnx",
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    bm25_built = bm25.build(
        root / "bm25.sqlite3",
        [
            SearchDocument(image_id=1, titles=("cat",)),
            SearchDocument(image_id=2, titles=("ordinary",)),
            SearchDocument(image_id=3, ocr_texts=("quokka exact phrase",)),
        ],
    )
    if corrupt_bm25:
        (root / "bm25.sqlite3").write_bytes(b"not sqlite")
    dense: list[search.DenseIndex] = []
    if with_bge:
        bge_index = faiss.IndexFlatIP(bge.DIMENSION)
        bge_index.add(np.array([[1.0, *([0.0] * 383)]], dtype=np.float32))
        faiss.write_index(bge_index, str(root / "bge_index.faiss"))
        (root / "bge_mapping.json").write_text(json.dumps({"0": 3}))
        (root / "bge_metadata.json").write_text(
            json.dumps(
                {
                    **metadata,
                    "embed_model": bge.SOURCE_MODEL,
                    "dimension": bge.DIMENSION,
                    "encoder_repo": bge.EXPORT.repo,
                    "encoder_revision": bge.EXPORT.revision,
                    "encoder_variant": bge.EXPORT.variant,
                    "kind": "bge",
                }
            )
        )
        dense_files = tuple(
            search.File(
                name=name,
                key=f"indexes/{version}/{name}",
                sha256="a" * 64,
            )
            for name in ("bge_index.faiss", "bge_mapping.json", "bge_metadata.json")
        )
        dense.append(
            search.DenseIndex(
                retriever="bge",
                model=bge.SOURCE_MODEL,
                dimension=bge.DIMENSION,
                encoder=search.Encoder(
                    repo=bge.EXPORT.repo,
                    revision=bge.EXPORT.revision,
                    variant=bge.EXPORT.variant,
                    threads=2,
                ),
                document_content_sha256="a" * 64,
                projection_version=1,
                render_version=1,
                count=1,
                files=dense_files,  # type: ignore[arg-type]
            )
        )
    return PreparedLoad(
        version=version,
        workspace=str(root),
        paths={path.name: str(path) for path in root.iterdir()},
        bm25=search.Bm25File(
            key=f"indexes/{version}/bm25.sqlite3",
            sha256=bm25_built.sha256,
            length=bm25_built.length,
            count=bm25_built.count,
            tokenizer=bm25.TOKENIZER,
            weights=bm25.WEIGHTS,
            sqlite_version=bm25_built.sqlite_version,
        ),
        dense=dense,
        encoder=search.Encoder(
            repo="test/encoder", revision=revision, variant="model.onnx", threads=1
        ),
    )


def _resident() -> Resident:
    return Resident(
        encoder_factory=lambda config: (
            _BgeEncoder() if config.repo == bge.EXPORT.repo else _Encoder()
        )
    )


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
    resident.load(_generation(tmp_path, "v1", with_bge=True))
    resident.switch("v1")

    with pytest.raises(search.Incompatible):
        resident.load(_generation(tmp_path, "v2", broken=True))

    assert resident.status().serving_version == "v1"
    assert resident.status().candidate_version is None


def test_corrupt_bm25_keeps_the_active_generation_serving(tmp_path: Path) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "v1"))
    resident.switch("v1")

    with pytest.raises(search.Incompatible, match="BM25 generation is invalid"):
        resident.load(_generation(tmp_path, "v2", corrupt_bm25=True))

    assert resident.status().serving_version == "v1"
    assert resident.status().bm25_available is True


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


def test_selected_recipe_survives_activation_and_previous_generation_rollback(
    tmp_path: Path,
) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "previous", with_bge=True))
    resident.switch("previous")
    resident.load(_generation(tmp_path, "selected", with_bge=True))
    resident.switch("selected")

    selected = resident.query(
        search.CandidateRequest(
            query=search.Query(text="quokka", recipe_id="image_bm25_bge"),
            count=10,
        )
    )
    assert selected.version == "selected"
    assert resident.status().retained_version == "previous"

    resident.rollback("selected")
    previous = resident.query(
        search.CandidateRequest(
            query=search.Query(text="quokka", recipe_id="image_bm25_bge"),
            count=10,
        )
    )
    assert previous.version == "previous"


def test_image_hybrid_and_similar_search_preserve_frozen_order(tmp_path: Path) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "v1", with_bge=True))
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
    assert [hit.image_id for hit in hybrid.candidates] == [1, 3, 2]
    assert [hit.image_id for hit in similar.candidates] == [2, 3]
    assert image.candidates[0].score == pytest.approx(1.0, abs=1e-6)


def test_cursor_returns_the_next_stable_candidate_batch(tmp_path: Path) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "v1", with_bge=True))
    resident.switch("v1")
    query = search.Query(text="cat")

    first = resident.query(search.CandidateRequest(query=query, count=2))
    second = resident.query(search.CandidateRequest(query=query, count=2, cursor=first.cursor))

    assert [hit.image_id for hit in first.candidates] == [1, 3]
    assert [hit.image_id for hit in second.candidates] == [2]
    assert second.exhausted is True


def test_hybrid_first_page_uses_fixed_recipe_depth(tmp_path: Path) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "v1", with_bge=True))
    resident.switch("v1")

    first = resident.query(
        search.CandidateRequest(
            query=search.Query(text="cat", mode="hybrid", limit=1),
            count=1,
        )
    )

    assert [hit.image_id for hit in first.candidates] == [1]


def test_bm25_recipe_promotes_an_exact_ocr_match(tmp_path: Path) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "v1"))
    resident.switch("v1")

    image = resident.query(
        search.CandidateRequest(
            query=search.Query(text="quokka", recipe_id="image_only"),
            count=2,
        )
    )
    result = resident.query(
        search.CandidateRequest(
            query=search.Query(text="quokka", recipe_id="image_bm25"),
            count=10,
        )
    )

    assert 3 not in [candidate.image_id for candidate in image.candidates]
    assert result.candidates[0].image_id == 3


def test_bge_recipe_warms_and_queries_its_independent_encoder(tmp_path: Path) -> None:
    resident = _resident()
    resident.load(_generation(tmp_path, "v1", with_bge=True))
    status = resident.switch("v1")

    result = resident.query(
        search.CandidateRequest(
            query=search.Query(text="semantic match", recipe_id="image_bge"),
            count=10,
        )
    )

    assert status.bge_available is True
    assert result.candidates[0].image_id == 3


def test_equivalent_generations_share_encoder_and_release_replaced_workspaces(
    tmp_path: Path,
) -> None:
    constructions = 0

    def factory(_: search.Encoder) -> _Encoder:
        nonlocal constructions
        constructions += 1
        return _Encoder()

    resident = Resident(encoder_factory=factory)
    first = _generation(tmp_path, "v1")
    second = _generation(tmp_path, "v2")
    third = _generation(tmp_path, "v3")

    resident.load(first)
    resident.switch("v1")
    resident.load(second)
    resident.switch("v2")
    resident.load(third)
    resident.switch("v3")

    assert constructions == 1
    assert not Path(first.workspace).exists()
    assert Path(second.workspace).exists()
    assert Path(third.workspace).exists()
    assert (Path(second.workspace) / "bm25.sqlite3").exists()
    assert (Path(third.workspace) / "bm25.sqlite3").exists()
    batch = resident.query(
        search.CandidateRequest(query=search.Query(text="cat", recipe_id="image_only"), count=1)
    )
    assert [candidate.image_id for candidate in batch.candidates] == [1]


def test_candidate_replacement_rollback_and_clear_release_each_workspace(
    tmp_path: Path,
) -> None:
    resident = _resident()
    first = _generation(tmp_path, "v1")
    replaced = _generation(tmp_path, "replaced")
    second = _generation(tmp_path, "v2")
    pending = _generation(tmp_path, "pending")

    resident.load(first)
    resident.load(replaced)
    assert not Path(first.workspace).exists()
    resident.switch("replaced")
    resident.load(second)
    resident.switch("v2")
    resident.load(pending)
    resident.rollback("v2")

    assert not Path(pending.workspace).exists()
    assert resident.status().serving_version == "replaced"
    resident.clear()
    assert not Path(replaced.workspace).exists()
    assert not Path(second.workspace).exists()
    assert resident.status().ready is False


def test_failed_load_releases_workspace_and_encoder_reference(tmp_path: Path) -> None:
    constructions = 0

    def factory(_: search.Encoder) -> _Encoder:
        nonlocal constructions
        constructions += 1
        return _Encoder()

    resident = Resident(encoder_factory=factory)
    broken = _generation(tmp_path, "v1", broken=True)
    with pytest.raises(search.Incompatible):
        resident.load(broken)
    assert not Path(broken.workspace).exists()

    valid = _generation(tmp_path, "v1")
    resident.load(valid)
    assert constructions == 1


def test_failed_encoder_construction_can_be_retried(tmp_path: Path) -> None:
    attempts = 0

    def factory(_: search.Encoder) -> _Encoder:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("construction failed")
        return _Encoder()

    resident = Resident(encoder_factory=factory)
    failed = _generation(tmp_path, "v1")
    with pytest.raises(search.Failed, match="text encoder load failed"):
        resident.load(failed)
    assert not Path(failed.workspace).exists()

    resident.load(_generation(tmp_path, "v1"))
    assert attempts == 2


def test_distinct_encoder_identity_constructs_a_new_session(tmp_path: Path) -> None:
    constructions = 0

    def factory(_: search.Encoder) -> _Encoder:
        nonlocal constructions
        constructions += 1
        return _Encoder()

    resident = Resident(encoder_factory=factory)
    resident.load(_generation(tmp_path, "v1"))
    resident.switch("v1")
    resident.load(_generation(tmp_path, "v2", revision="rev-2"))

    assert constructions == 2


def test_lifecycle_events_have_numeric_resources_and_no_query_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class Logger:
        def info(self, event: str, **fields: object) -> None:
            events.append((event, fields))

    monkeypatch.setattr("mimeme.compute.search.structlog.get_logger", lambda: Logger())
    resident = _resident()
    resident.load(_generation(tmp_path, "v1"))
    resident.switch("v1")
    resident.load(_generation(tmp_path, "v2"))
    resident.switch("v2")
    resident.rollback("v2")
    resident.clear()

    assert {event for event, _ in events} >= {
        "search_generation_load",
        "search_generation_switch",
        "search_generation_rollback",
        "search_generation_evicted",
    }
    for _, fields in events:
        assert isinstance(fields["duration_ms"], (int, float))
        assert fields["duration_ms"] >= 0
        assert isinstance(fields["rss_bytes"], int)
        assert fields["rss_bytes"] >= 0
        assert fields["generation_version"]
        assert fields["encoder_revision"]
        assert "query" not in fields
