from __future__ import annotations

import json

import faiss  # type: ignore[import-untyped]
import numpy as np

from mimeme import index
from mimeme.compute.index import build, pack
from mimeme.compute.model import ChildOk
from mimeme.compute.protocol import parse_reply
from mimeme.compute.supervisor import Supervisor


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


def test_a_shard_build_matches_the_individual_object_build(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    vectors = [
        np.array([3.0, 0.0], dtype=np.float32),
        np.array([0.0, 4.0], dtype=np.float32),
    ]
    for position, vector in enumerate(vectors, start=1):
        np.save(inputs / f"{position}.npy", vector)
        np.save(inputs / f"{position}_text.npy", vector[::-1].copy())
    np.save(inputs / "shard-image.npy", np.stack(vectors))
    np.save(inputs / "shard-text.npy", np.stack([vector[::-1] for vector in vectors]))

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
                        text_path=str(inputs / "shard-text.npy"),
                    )
                ]
                if sealed
                else []
            ),
            embeddings=[
                index.LocalEmbedding(
                    image_id=position, shard=0, row=position - 1, text_present=True
                )
                if sealed
                else index.LocalEmbedding(
                    image_id=position,
                    image_path=str(inputs / f"{position}.npy"),
                    text_path=str(inputs / f"{position}_text.npy"),
                )
                for position in (1, 2)
            ],
        )

    loose = build(request(False, str(tmp_path / "loose")))
    sealed = build(request(True, str(tmp_path / "sealed")))

    assert loose.image_count == sealed.image_count == 2
    assert loose.text_count == sealed.text_count == 2
    assert {file.name: file.sha256 for file in loose.files} == {
        file.name: file.sha256 for file in sealed.files
    }


def test_pack_writes_paired_matrices_and_zeroes_a_missing_text_row(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.save(inputs / "1.npy", np.array([3.0, 0.0], dtype=np.float32))
    np.save(inputs / "1_text.npy", np.array([0.0, 3.0], dtype=np.float32))
    np.save(inputs / "2.npy", np.array([0.0, 4.0], dtype=np.float32))

    packed = pack(
        index.PackCall(
            members=[
                index.LocalMember(
                    image_id=1,
                    image_path=str(inputs / "1.npy"),
                    text_path=str(inputs / "1_text.npy"),
                ),
                index.LocalMember(image_id=2, image_path=str(inputs / "2.npy")),
            ],
            image_out=str(tmp_path / "image.npy"),
            text_out=str(tmp_path / "text.npy"),
        )
    )

    images = np.load(tmp_path / "image.npy")
    texts = np.load(tmp_path / "text.npy")
    assert (packed.rows, packed.dimension) == (2, 2)
    assert images.tolist() == [[3.0, 0.0], [0.0, 4.0]]
    assert texts.tolist() == [[0.0, 3.0], [0.0, 0.0]]
    assert packed.image.length == (tmp_path / "image.npy").stat().st_size


async def test_the_spawned_index_child_serves_both_build_and_pack(tmp_path) -> None:
    np.save(tmp_path / "1.npy", np.array([3.0, 0.0], dtype=np.float32))
    supervisor = Supervisor(tmp_path / "sock")
    await supervisor.start(roles=("index",))
    try:
        call = index.PackCall(
            members=[index.LocalMember(image_id=1, image_path=str(tmp_path / "1.npy"))],
            image_out=str(tmp_path / "image.npy"),
            text_out=str(tmp_path / "text.npy"),
        )
        reply = parse_reply(await supervisor.call("index", call.model_dump_json().encode()))
        assert isinstance(reply, ChildOk)
        packed = index.Packed.model_validate(reply.result)
        assert (packed.rows, packed.dimension) == (1, 2)
        assert np.load(tmp_path / "image.npy").tolist() == [[3.0, 0.0]]
    finally:
        await supervisor.close()
