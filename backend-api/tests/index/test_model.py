from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mimeme import index


def test_build_contract_is_bounded_to_object_references() -> None:
    build = index.Build(
        job_id="rebuild-1",
        version="v2-g7-rebuild-1",
        target_generation=7,
        model="test/embed",
        index_type="flat",
        dimension=2,
        encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
        embeddings=[
            index.Embedding(image_id=10, image_key="embeddings/10.npy", text_key=None),
            index.Embedding(
                image_id=11, image_key="embeddings/11.npy", text_key="embeddings/11_text.npy"
            ),
        ],
    )

    payload = build.model_dump_json()

    assert len(payload) < 1_000
    assert "embeddings/10.npy" in payload
    assert "[[" not in payload
    assert index.Build.model_validate_json(payload) == build


def test_complete_manifest_requires_generation_artifacts_and_own_prefix() -> None:
    files = [
        index.File(name="index.faiss", key="indexes/v2/index.faiss", sha256="0" * 64, length=4),
        index.File(name="mapping.json", key="indexes/v2/mapping.json", sha256="1" * 64, length=4),
        index.File(name="metadata.json", key="indexes/v2/metadata.json", sha256="2" * 64, length=4),
    ]

    manifest = index.Manifest(
        version="v2",
        target_generation=2,
        model="test/embed",
        index_type="flat",
        encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
        dimension=2,
        image_count=1,
        files=files,
        complete_key="indexes/v2/complete.json",
    )
    assert json.loads(manifest.model_dump_json())["version"] == "v2"

    with pytest.raises(ValidationError, match="generation prefix"):
        index.Manifest(
            version="v2",
            target_generation=2,
            model="test/embed",
            index_type="flat",
            encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
            dimension=2,
            image_count=1,
            files=[files[0].model_copy(update={"key": "indexes/other/index.faiss"}), *files[1:]],
            complete_key="indexes/v2/complete.json",
        )


def test_manifest_v2_requires_matching_document_artifact() -> None:
    files = [
        index.File(name="index.faiss", key="indexes/v2/index.faiss", sha256="0" * 64, length=4),
        index.File(name="mapping.json", key="indexes/v2/mapping.json", sha256="1" * 64, length=4),
        index.File(name="metadata.json", key="indexes/v2/metadata.json", sha256="2" * 64, length=4),
    ]
    document = index.DocumentFile(
        key="indexes/v2/documents.jsonl.zst",
        sha256="3" * 64,
        content_sha256="4" * 64,
        length=10,
        count=1,
    )

    manifest = index.Manifest(
        format_version=2,
        version="v2",
        target_generation=2,
        model="test/embed",
        index_type="flat",
        encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
        dimension=2,
        image_count=1,
        files=files,
        documents=document,
        complete_key="indexes/v2/complete.json",
    )

    assert index.Manifest.model_validate_json(manifest.model_dump_json()) == manifest
    with pytest.raises(ValidationError, match="requires a document artifact"):
        index.Manifest(
            format_version=2,
            version="v2",
            target_generation=2,
            model="test/embed",
            index_type="flat",
            encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
            dimension=2,
            image_count=1,
            files=files,
            complete_key="indexes/v2/complete.json",
        )


def test_manifest_bm25_matches_documents_and_generation() -> None:
    files = [
        index.File(name="index.faiss", key="indexes/v2/index.faiss", sha256="0" * 64, length=4),
        index.File(name="mapping.json", key="indexes/v2/mapping.json", sha256="1" * 64, length=4),
        index.File(name="metadata.json", key="indexes/v2/metadata.json", sha256="2" * 64, length=4),
    ]
    document = index.DocumentFile(
        key="indexes/v2/documents.jsonl.zst",
        sha256="3" * 64,
        content_sha256="4" * 64,
        length=10,
        count=1,
    )
    bm25 = index.Bm25File(
        key="indexes/v2/bm25.sqlite3",
        sha256="5" * 64,
        length=4096,
        count=1,
        weights=(4, 4, 4, 2, 2, 2, 1),
        sqlite_version="3.40.1",
    )
    values = {
        "format_version": 2,
        "version": "v2",
        "target_generation": 2,
        "model": "test/embed",
        "index_type": "flat",
        "encoder": index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
        "dimension": 2,
        "image_count": 1,
        "files": files,
        "documents": document,
        "bm25": bm25,
        "complete_key": "indexes/v2/complete.json",
    }

    manifest = index.Manifest.model_validate(values)
    assert manifest.bm25 == bm25

    with pytest.raises(ValidationError, match="BM25 document count must match image count"):
        index.Manifest.model_validate({**values, "bm25": bm25.model_copy(update={"count": 2})})
    with pytest.raises(ValidationError, match="BM25 artifact must use its generation prefix"):
        index.Manifest.model_validate(
            {**values, "bm25": bm25.model_copy(update={"key": "indexes/v3/bm25.sqlite3"})}
        )
    with pytest.raises(ValidationError, match="BM25 field weights are incompatible"):
        index.Bm25File.model_validate({**bm25.model_dump(), "weights": [1] * 7})


def test_state_machine_rejects_invalid_activation_transition() -> None:
    state = index.State(job_id="rebuild-1")

    with pytest.raises(index.InvalidTransition, match="new to active"):
        index.transition(state, index.Phase.ACTIVE)

    prepared = index.transition(state, index.Phase.PREPARED)
    built = index.transition(prepared, index.Phase.BUILT)
    active = index.transition(built, index.Phase.ACTIVE)
    assert active.phase is index.Phase.ACTIVE
