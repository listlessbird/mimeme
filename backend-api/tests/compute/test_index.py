from __future__ import annotations

import json

import faiss  # type: ignore[import-untyped]
import numpy as np

from mimeme import index
from mimeme.compute.index import build, pack
from mimeme.compute.model import ChildOk
from mimeme.compute.protocol import parse_reply
from mimeme.compute.supervisor import Supervisor
from mimeme.inference import bge


def test_native_builder_writes_retrievable_normalized_generation(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    output = tmp_path / "output"
    inputs.mkdir()
    np.save(inputs / "1.npy", np.array([3.0, 0.0], dtype=np.float32))
    np.save(inputs / "2.npy", np.array([0.0, 4.0], dtype=np.float32))

    result = build(
        index.PreparedBuild(
            version="v2-test",
            target_generation=4,
            model="test/embed",
            index_type="flat",
            dimension=2,
            encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
            output_dir=str(output),
            embeddings=[
                index.LocalEmbedding(image_id=1, image_path=str(inputs / "1.npy")),
                index.LocalEmbedding(image_id=2, image_path=str(inputs / "2.npy")),
            ],
        )
    )

    built = faiss.read_index(str(output / "index.faiss"))
    scores, rows = built.search(np.array([[1.0, 0.0]], dtype=np.float32), 2)
    mapping = json.loads((output / "mapping.json").read_text())
    assert mapping[str(rows[0][0])] == 1
    assert scores[0][0] == 1.0
    assert {file.name for file in result.files} == {
        "index.faiss",
        "mapping.json",
        "metadata.json",
    }


def test_native_builder_hosts_independent_384d_bge_and_siglip_indexes(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    output = tmp_path / "output"
    inputs.mkdir()
    np.save(inputs / "siglip.npy", np.array([1.0, 0.0], dtype=np.float32))
    np.save(inputs / "bge.npy", np.eye(1, bge.DIMENSION, dtype=np.float32))
    (inputs / "bge-mapping.json").write_text('{"0": 7}')
    (inputs / "bge-meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "retriever": "bge",
                "version": "v2-test",
                "model": bge.SOURCE_MODEL,
                "dimension": bge.DIMENSION,
                "dtype": "float32",
                "normalized": True,
                "count": 1,
                "document_content_sha256": "a" * 64,
                "projection_version": 1,
                "render_version": 1,
                "export": bge.EXPORT.model_dump(mode="json"),
            }
        )
    )
    encoder = index.Encoder(
        repo=bge.EXPORT.repo,
        revision=bge.EXPORT.revision,
        variant=bge.EXPORT.variant,
        threads=2,
    )

    result = build(
        index.PreparedBuild(
            version="v2-test",
            target_generation=4,
            model="test/embed",
            index_type="flat",
            dimension=2,
            encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
            output_dir=str(output),
            embeddings=[index.LocalEmbedding(image_id=7, image_path=str(inputs / "siglip.npy"))],
            dense=[
                index.LocalDense(
                    retriever="bge",
                    version="v2-test",
                    model=bge.SOURCE_MODEL,
                    dimension=bge.DIMENSION,
                    encoder=encoder,
                    document_content_sha256="a" * 64,
                    projection_version=1,
                    render_version=1,
                    count=1,
                    vectors_path=str(inputs / "bge.npy"),
                    mapping_path=str(inputs / "bge-mapping.json"),
                    metadata_path=str(inputs / "bge-meta.json"),
                )
            ],
        )
    )

    assert faiss.read_index(str(output / "index.faiss")).d == 2
    assert faiss.read_index(str(output / "bge_index.faiss")).d == bge.DIMENSION
    assert result.dense_counts == {"bge": 1}
    assert {file.name for file in result.files}.issuperset(
        {"bge_index.faiss", "bge_mapping.json", "bge_metadata.json"}
    )


def test_a_shard_build_matches_the_individual_object_build(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    vectors = [
        np.array([3.0, 0.0], dtype=np.float32),
        np.array([0.0, 4.0], dtype=np.float32),
    ]
    for position, vector in enumerate(vectors, start=1):
        np.save(inputs / f"{position}.npy", vector)
    np.save(inputs / "shard-image.npy", np.stack(vectors))

    def request(sealed: bool, output: str) -> index.PreparedBuild:
        return index.PreparedBuild(
            version="v2-test",
            target_generation=4,
            model="test/embed",
            index_type="flat",
            dimension=2,
            encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
            output_dir=output,
            shards=(
                [
                    index.LocalShard(
                        number=0,
                        image_path=str(inputs / "shard-image.npy"),
                    )
                ]
                if sealed
                else []
            ),
            embeddings=[
                index.LocalEmbedding(image_id=position, shard=0, row=position - 1)
                if sealed
                else index.LocalEmbedding(
                    image_id=position,
                    image_path=str(inputs / f"{position}.npy"),
                )
                for position in (1, 2)
            ],
        )

    loose = build(request(False, str(tmp_path / "loose")))
    sealed = build(request(True, str(tmp_path / "sealed")))

    assert loose.image_count == sealed.image_count == 2
    assert {file.name: file.sha256 for file in loose.files} == {
        file.name: file.sha256 for file in sealed.files
    }


def test_pack_writes_the_image_matrix(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.save(inputs / "1.npy", np.array([3.0, 0.0], dtype=np.float32))
    np.save(inputs / "2.npy", np.array([0.0, 4.0], dtype=np.float32))

    packed = pack(
        index.PackCall(
            members=[
                index.LocalMember(image_id=1, image_path=str(inputs / "1.npy")),
                index.LocalMember(image_id=2, image_path=str(inputs / "2.npy")),
            ],
            image_out=str(tmp_path / "image.npy"),
        )
    )

    images = np.load(tmp_path / "image.npy")
    assert (packed.rows, packed.dimension) == (2, 2)
    assert images.tolist() == [[3.0, 0.0], [0.0, 4.0]]
    assert packed.image.length == (tmp_path / "image.npy").stat().st_size


def test_pack_upcasts_half_precision_embeddings(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.save(inputs / "1.npy", np.array([3.0, 0.0], dtype=np.float16))
    np.save(inputs / "2.npy", np.array([0.0, 4.0], dtype=np.float32))

    packed = pack(
        index.PackCall(
            members=[
                index.LocalMember(image_id=1, image_path=str(inputs / "1.npy")),
                index.LocalMember(image_id=2, image_path=str(inputs / "2.npy")),
            ],
            image_out=str(tmp_path / "image.npy"),
        )
    )

    images = np.load(tmp_path / "image.npy")
    assert (packed.rows, packed.dimension) == (2, 2)
    assert images.dtype == np.float32
    assert images.tolist() == [[3.0, 0.0], [0.0, 4.0]]


async def test_the_spawned_index_child_serves_both_build_and_pack(tmp_path) -> None:
    np.save(tmp_path / "1.npy", np.array([3.0, 0.0], dtype=np.float32))
    supervisor = Supervisor(tmp_path / "sock")
    await supervisor.start(roles=("index",))
    try:
        call = index.PackCall(
            members=[index.LocalMember(image_id=1, image_path=str(tmp_path / "1.npy"))],
            image_out=str(tmp_path / "image.npy"),
        )
        reply = parse_reply(await supervisor.call("index", call.model_dump_json().encode()))
        assert isinstance(reply, ChildOk)
        packed = index.Packed.model_validate(reply.result)
        assert (packed.rows, packed.dimension) == (1, 2)
        assert np.load(tmp_path / "image.npy").tolist() == [[3.0, 0.0]]
    finally:
        await supervisor.close()
