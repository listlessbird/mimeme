from __future__ import annotations

import io

import numpy as np
from tests.support.storage import Memory

from mimeme import index, storage
from mimeme.compute.index import build
from mimeme.compute.model import ChildOk
from mimeme.index import pack
from mimeme.index.gateway import Gateway


class _Calls:
    async def call(self, role: str, request: bytes) -> bytes:
        assert role == "index"
        call = index.BuildCall.model_validate_json(request)
        return ChildOk(result=build(call.build).model_dump()).model_dump_json().encode()


class _ObservedMemory(Memory):
    def __init__(self) -> None:
        super().__init__()
        self.writes: list[str] = []

    async def put(self, obj, body, **kwargs):  # noqa: ANN001, ANN003, ANN202
        self.writes.append(obj.key)
        return await super().put(obj, body, **kwargs)

    async def put_bytes(self, obj, data, **kwargs):  # noqa: ANN001, ANN003, ANN202
        self.writes.append(obj.key)
        return await super().put_bytes(obj, data, **kwargs)


def _npy(vector: list[float]) -> bytes:
    output = io.BytesIO()
    np.save(output, np.array(vector, dtype=np.float32))
    return output.getvalue()


async def test_gateway_owns_transfer_and_publishes_completeness_last(tmp_path) -> None:
    artifacts = _ObservedMemory()
    await artifacts.put_bytes(
        storage.Object("embeddings/1.npy"), _npy([1, 0]), content_type="application/octet-stream"
    )
    artifacts.writes.clear()
    gateway = Gateway(_Calls(), artifacts=artifacts, workspace_dir=tmp_path)

    result = await gateway.build(
        index.Build(
            job_id="rebuild-1",
            version="v2-g3-rebuild-1",
            target_generation=3,
            model="test/embed",
            index_type="flat",
            dimension=2,
            encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
            embeddings=[index.Embedding(image_id=1, image_key="embeddings/1.npy")],
        )
    )

    assert result.manifest is not None
    assert artifacts.writes[-1] == result.manifest.complete_key
    assert await artifacts.stat(storage.Object(result.manifest.complete_key)) is not None
    assert not list(tmp_path.iterdir())


class _CountingMemory(Memory):
    def __init__(self) -> None:
        super().__init__()
        self.reads: list[str] = []
        self.probes: list[str] = []

    def read(self, obj):  # noqa: ANN001, ANN202
        self.reads.append(obj.key)
        return super().read(obj)

    async def read_bytes(self, obj, **kwargs):  # noqa: ANN001, ANN003, ANN202
        self.reads.append(obj.key)
        return await super().read_bytes(obj, **kwargs)

    async def stat(self, obj):  # noqa: ANN001, ANN202
        self.probes.append(obj.key)
        return await super().stat(obj)


async def test_a_rebuild_reads_each_referenced_vector_once_and_probes_nothing(tmp_path) -> None:
    artifacts = _CountingMemory()
    for image_id in (1, 2):
        await artifacts.put_bytes(
            storage.Object(f"embeddings/{image_id}.npy"),
            _npy([1, 0]),
            content_type="application/octet-stream",
        )
    await artifacts.put_bytes(
        storage.Object("embeddings/1_text.npy"),
        _npy([0, 1]),
        content_type="application/octet-stream",
    )
    gateway = Gateway(_Calls(), artifacts=artifacts, workspace_dir=tmp_path)
    request = index.Build(
        job_id="rebuild-2",
        version="v2-g4-rebuild-2",
        target_generation=4,
        model="test/embed",
        index_type="flat",
        dimension=2,
        encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
        embeddings=[
            index.Embedding(
                image_id=1, image_key="embeddings/1.npy", text_key="embeddings/1_text.npy"
            ),
            index.Embedding(image_id=2, image_key="embeddings/2.npy"),
        ],
        planned_reads=3,
    )

    result = await gateway.build(request)

    assert result.outcome == "built"
    assert artifacts.reads == [
        "embeddings/1.npy",
        "embeddings/1_text.npy",
        "embeddings/2.npy",
    ]
    assert artifacts.probes == []
    assert len(artifacts.reads) == request.planned_reads


async def _shard(artifacts: Memory, key: str, rows: list[list[float]]) -> None:
    output = io.BytesIO()
    np.save(output, np.array(rows, dtype=np.float32))
    await artifacts.put_bytes(
        storage.Object(key), output.getvalue(), content_type="application/octet-stream"
    )


async def test_a_mixed_corpus_reads_one_object_per_shard_plus_the_unsealed_tail(
    tmp_path,
) -> None:
    artifacts = _CountingMemory()
    await _shard(artifacts, pack.locate("test/embed", 0, text=False), [[1, 0], [0, 1], [1, 1]])
    await _shard(artifacts, pack.locate("test/embed", 0, text=True), [[0, 1], [0, 0], [1, 0]])
    await _shard(artifacts, pack.locate("test/embed", 1, text=False), [[1, 2]])
    await _shard(artifacts, pack.locate("test/embed", 1, text=True), [[0, 0]])
    await artifacts.put_bytes(
        storage.Object("embeddings/tail.npy"), _npy([2, 1]), content_type="application/octet-stream"
    )
    gateway = Gateway(_Calls(), artifacts=artifacts, workspace_dir=tmp_path)
    embeddings = [
        index.Embedding(image_id=1, shard=0, row=0, text_present=True),
        index.Embedding(image_id=2, shard=0, row=1, text_present=False),
        index.Embedding(image_id=3, shard=0, row=2, text_present=True),
        index.Embedding(image_id=4, shard=1, row=0, text_present=False),
        index.Embedding(image_id=5, image_key="embeddings/tail.npy"),
    ]
    request = index.Build(
        job_id="rebuild-3",
        version="v2-g5-rebuild-3",
        target_generation=5,
        model="test/embed",
        index_type="flat",
        dimension=2,
        encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
        embeddings=embeddings,
        planned_reads=pack.reads(embeddings),
    )

    result = await gateway.build(request)

    assert result.manifest is not None
    assert result.manifest.image_count == 5
    assert result.manifest.text_count == 2
    assert artifacts.probes == []
    assert artifacts.reads == [
        pack.locate("test/embed", 0, text=False),
        pack.locate("test/embed", 0, text=True),
        pack.locate("test/embed", 1, text=False),
        "embeddings/tail.npy",
    ]
    assert len(artifacts.reads) == request.planned_reads == 4
