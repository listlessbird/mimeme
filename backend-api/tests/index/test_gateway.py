from __future__ import annotations

import io

import numpy as np
from tests.support.storage import Memory

from mimeme import index, storage
from mimeme.compute.index import build
from mimeme.compute.model import ChildOk
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
