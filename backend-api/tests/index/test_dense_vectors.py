from __future__ import annotations

import io
import json

import numpy as np
from tests.support.storage import Memory

from mimeme import inference, storage
from mimeme.index import dense_vectors, documents
from mimeme.search.document import SearchDocument


class _Remote:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.calls = 0
        self.fail_once = fail_once

    async def embed_bge(
        self,
        batch: inference.bge.EncodeBatch,
        *,
        progress=None,  # noqa: ANN001
    ) -> inference.bge.EncodedBatch:
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise inference.Unavailable("temporary remote failure")
        return inference.bge.EncodedBatch(
            document_content_sha256=batch.document_content_sha256,
            export=batch.export,
            items=tuple(
                inference.bge.Vector(
                    image_id=item.image_id,
                    values=(1.0, *([0.0] * (inference.bge.DIMENSION - 1))),
                )
                for item in batch.items
            ),
        )


async def test_remote_bge_vectors_are_generation_owned_and_checksum_tied(monkeypatch) -> None:
    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(dense_vectors.asyncio, "sleep", no_wait)
    artifacts = Memory()
    document_file = await documents.publish(
        artifacts,
        version="v2-g1-test",
        documents=[
            SearchDocument(image_id=1, titles=("Cat",)),
            SearchDocument(image_id=2, ocr_texts=("deploy failed",)),
        ],
    )
    remote = _Remote(fail_once=True)

    result = await dense_vectors.encode_bge(
        artifacts,
        remote,  # type: ignore[arg-type]
        version="v2-g1-test",
        document_file=document_file,
    )

    assert remote.calls == 2
    assert result.document_content_sha256 == document_file.content_sha256
    assert result.encoder.revision == inference.bge.EXPORT.revision
    raw_vectors = await artifacts.read_bytes(
        storage.Object(result.vectors.key), max_bytes=result.vectors.length
    )
    matrix = np.load(io.BytesIO(raw_vectors), allow_pickle=False)
    assert matrix.shape == (2, inference.bge.DIMENSION)
    assert matrix.dtype == np.float32
    mapping = json.loads(
        await artifacts.read_bytes(
            storage.Object(result.mapping.key), max_bytes=result.mapping.length
        )
    )
    assert mapping == {"0": 1, "1": 2}
    metadata = json.loads(
        await artifacts.read_bytes(
            storage.Object(result.metadata.key), max_bytes=result.metadata.length
        )
    )
    assert metadata["document_content_sha256"] == document_file.content_sha256
    assert metadata["export"]["revision"] == inference.bge.EXPORT.revision
