from __future__ import annotations

import asyncio
import hashlib
import io
import json
from collections.abc import Awaitable, Callable

import numpy as np

from mimeme import inference, storage
from mimeme.index import documents
from mimeme.index.model import Blob, DenseVectors, DocumentFile, Encoder

Progress = Callable[[str, float], Awaitable[None]]
_BATCH_SIZE = 128
_REMOTE_ATTEMPTS = 3


async def encode_bge(
    artifacts: storage.Store,
    client: inference.Client,
    *,
    version: str,
    document_file: DocumentFile,
    progress: Progress | None = None,
    batch_size: int = _BATCH_SIZE,
    encoder_threads: int = 2,
) -> DenseVectors:
    values = await documents.verify(artifacts, document_file)
    vectors: list[np.ndarray] = []
    total = len(values)
    for offset in range(0, total, batch_size):
        selected = values[offset : offset + batch_size]
        request = inference.bge.EncodeBatch(
            document_content_sha256=document_file.content_sha256,
            projection_version=document_file.projection_version,
            export=inference.bge.EXPORT,
            items=tuple(
                inference.bge.CorpusItem(
                    image_id=item.image_id,
                    text=inference.bge.render_document(item),
                )
                for item in selected
            ),
        )
        result = await _encode_with_retry(client, request)
        inference.bge.validate_result(request, result)
        vectors.extend(np.asarray(item.values, dtype=np.float32) for item in result.items)
        if progress is not None:
            await progress("remote", (offset + len(selected)) / max(total, 1))

    matrix = (
        np.stack(vectors).astype(np.float32, copy=False)
        if vectors
        else np.empty((0, inference.bge.DIMENSION), dtype=np.float32)
    )
    if matrix.shape != (total, inference.bge.DIMENSION):
        raise ValueError("BGE vector matrix shape does not match the document snapshot")
    image_ids = [item.image_id for item in values]
    vector_bytes = _npy(matrix)
    mapping_bytes = json.dumps(
        {str(row): image_id for row, image_id in enumerate(image_ids)},
        sort_keys=True,
    ).encode()
    metadata_bytes = json.dumps(
        {
            "schema_version": 1,
            "retriever": "bge",
            "version": version,
            "model": inference.bge.SOURCE_MODEL,
            "dimension": inference.bge.DIMENSION,
            "dtype": "float32",
            "normalized": True,
            "count": total,
            "document_content_sha256": document_file.content_sha256,
            "projection_version": document_file.projection_version,
            "render_version": inference.bge.RENDER_VERSION,
            "export": inference.bge.EXPORT.model_dump(mode="json"),
        },
        sort_keys=True,
    ).encode()
    blobs = (
        await _publish(artifacts, version, "bge_vectors.npy", vector_bytes),
        await _publish(
            artifacts,
            version,
            "bge_vectors_mapping.json",
            mapping_bytes,
            content_type="application/json",
        ),
        await _publish(
            artifacts,
            version,
            "bge_vectors_metadata.json",
            metadata_bytes,
            content_type="application/json",
        ),
    )
    return DenseVectors(
        version=version,
        encoder=Encoder(
            repo=inference.bge.EXPORT.repo,
            revision=inference.bge.EXPORT.revision,
            variant=inference.bge.EXPORT.variant,
            threads=encoder_threads,
        ),
        document_content_sha256=document_file.content_sha256,
        projection_version=document_file.projection_version,
        render_version=inference.bge.RENDER_VERSION,
        count=total,
        vectors=blobs[0],
        mapping=blobs[1],
        metadata=blobs[2],
    )


async def _encode_with_retry(
    client: inference.Client,
    request: inference.bge.EncodeBatch,
) -> inference.bge.EncodedBatch:
    for attempt in range(_REMOTE_ATTEMPTS):
        try:
            return await client.embed_bge(request)
        except (inference.Unavailable, inference.Timeout):
            if attempt == _REMOTE_ATTEMPTS - 1:
                raise
            await asyncio.sleep(2**attempt)
    raise AssertionError("BGE remote retry loop exhausted")


def _npy(matrix: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    return buffer.getvalue()


async def _publish(
    artifacts: storage.Store,
    version: str,
    name: str,
    data: bytes,
    *,
    content_type: str = "application/octet-stream",
) -> Blob:
    descriptor = Blob(
        name=name,
        key=f"indexes/{version}/{name}",
        sha256=hashlib.sha256(data).hexdigest(),
        length=len(data),
    )

    async def chunks():  # noqa: ANN202
        view = memoryview(data)
        for offset in range(0, len(view), 1024 * 1024):
            yield bytes(view[offset : offset + 1024 * 1024])

    await artifacts.put(
        storage.Object(descriptor.key),
        chunks(),
        length=descriptor.length,
        content_type=content_type,
        checksum=storage.Checksum(value=descriptor.sha256),
    )
    return descriptor
