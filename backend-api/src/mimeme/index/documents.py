from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import zstandard
from sqlalchemy import Row, select
from sqlalchemy.ext.asyncio import AsyncSession

from mimeme import storage
from mimeme.db.schema import (
    Annotation,
    EmbeddingShard,
    IngestURL,
    Processing,
    ProcessingStatus,
    SourceItem,
)
from mimeme.index.model import DocumentFile, Embedding, Snapshot
from mimeme.search import document

_MAX_COMPRESSED_BYTES = 1024 * 1024 * 1024


async def publish(
    artifacts: storage.Store,
    *,
    version: str,
    documents: list[document.SearchDocument],
) -> DocumentFile:
    content = b"".join(document.canonical_json(item).encode() + b"\n" for item in documents)
    compressed = zstandard.ZstdCompressor().compress(content)
    descriptor = DocumentFile(
        key=f"indexes/{version}/documents.jsonl.zst",
        sha256=hashlib.sha256(compressed).hexdigest(),
        content_sha256=hashlib.sha256(content).hexdigest(),
        length=len(compressed),
        count=len(documents),
        projection_version=document.PROJECTION_VERSION,
    )

    async def chunks():  # noqa: ANN202
        yield compressed

    await artifacts.put(
        storage.Object(descriptor.key),
        chunks(),
        length=descriptor.length,
        content_type="application/zstd",
        checksum=storage.Checksum(value=descriptor.sha256),
    )
    return descriptor


async def verify(
    artifacts: storage.Store,
    descriptor: DocumentFile,
) -> list[document.SearchDocument]:
    compressed = await artifacts.read_bytes(
        storage.Object(descriptor.key), max_bytes=_MAX_COMPRESSED_BYTES
    )
    if len(compressed) != descriptor.length:
        raise ValueError(f"document artifact has wrong length: {descriptor.key}")
    if hashlib.sha256(compressed).hexdigest() != descriptor.sha256:
        raise ValueError(f"document artifact checksum mismatch: {descriptor.key}")
    try:
        content = zstandard.ZstdDecompressor().decompress(compressed)
    except zstandard.ZstdError as exc:
        raise ValueError(f"document artifact is corrupt: {descriptor.key}") from exc
    if hashlib.sha256(content).hexdigest() != descriptor.content_sha256:
        raise ValueError(f"document content hash mismatch: {descriptor.key}")

    try:
        values = [
            document.SearchDocument.model_validate_json(line)
            for line in content.splitlines()
            if line
        ]
    except ValueError as exc:
        raise ValueError(f"document artifact contains invalid JSONL: {descriptor.key}") from exc
    if len(values) != descriptor.count:
        raise ValueError(f"document artifact count mismatch: {descriptor.key}")
    if any(value.projection_version != descriptor.projection_version for value in values):
        raise ValueError(f"document projection version mismatch: {descriptor.key}")
    if [value.image_id for value in values] != sorted(value.image_id for value in values):
        raise ValueError(f"document artifact is not ordered by image ID: {descriptor.key}")
    return values


async def capture(
    session: AsyncSession,
    *,
    model: str,
    target_generation: int,
) -> Snapshot:
    rows = (
        await session.execute(
            select(
                Processing.image_id,
                Processing.embed_s3_key,
                Processing.embed_dim,
                Processing.embed_shard,
                Processing.embed_row,
                EmbeddingShard.seq,
                Annotation.caption_text,
                Annotation.ocr_text,
                SourceItem.id.label("source_item_id"),
                SourceItem.title.label("source_item_title"),
                SourceItem.known_facts,
            )
            .outerjoin(
                EmbeddingShard,
                (EmbeddingShard.embed_model == Processing.embed_model)
                & (EmbeddingShard.number == Processing.embed_shard),
            )
            .outerjoin(Annotation, Annotation.image_id == Processing.image_id)
            .outerjoin(IngestURL, IngestURL.image_id == Processing.image_id)
            .outerjoin(SourceItem, SourceItem.id == IngestURL.source_item_id)
            .where(
                Processing.embed_status == ProcessingStatus.DONE,
                Processing.embed_model == model,
                Processing.embed_s3_key.is_not(None),
                Processing.embed_s3_key != "",
            )
            .order_by(Processing.image_id, SourceItem.id)
        )
    ).all()
    dimensions = {int(row.embed_dim) for row in rows if row.embed_dim is not None}
    if len(dimensions) > 1:
        raise ValueError(f"snapshot contains mixed embedding dimensions: {dimensions}")
    if rows and not dimensions:
        raise ValueError("snapshot embeddings have no recorded dimension")

    grouped: dict[int, list[Row[Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row.image_id), []).append(row)

    embeddings: list[Embedding] = []
    projected: list[document.SearchDocument] = []
    for image_id, image_rows in grouped.items():
        first = image_rows[0]
        embeddings.append(_embedding(first))
        projected.append(
            document.project(
                image_id,
                sources=(
                    _source_facts(row) for row in image_rows if row.source_item_id is not None
                ),
                caption=first.caption_text,
                ocr_text=first.ocr_text,
            )
        )
    return Snapshot(
        target_generation=target_generation,
        dimension=next(iter(dimensions), 0),
        embeddings=embeddings,
        documents=projected,
    )


def _embedding(row: Row[Any]) -> Embedding:
    if row.embed_shard is not None and row.embed_row is not None and row.seq is not None:
        return Embedding(
            image_id=row.image_id,
            shard=row.embed_shard,
            row=row.embed_row,
            seq=row.seq,
        )
    image_key = str(row.embed_s3_key)
    return Embedding(image_id=row.image_id, image_key=image_key)


def _source_facts(row: Row[Any]) -> document.SourceFacts:
    facts = row.known_facts if isinstance(row.known_facts, Mapping) else {}
    return document.source_facts(row.source_item_title, facts)
