from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mimeme.db.schema import (
    Annotation,
    Image,
    IngestURL,
    Processing,
    ProcessingStatus,
)
from mimeme.ingest.facts import Facts
from mimeme.ingest.rule import DEDUP_LOCK_KEY

_PHASH_THRESHOLD = 8


class ExistingImage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: int
    s3_key: str
    sha256: str
    width: int | None
    height: int | None
    format: str | None
    needs_annotation: bool
    needs_embedding: bool
    existing_caption: str | None
    existing_ocr_text: str | None


def _phash_to_uint64(hex_str: str | None) -> int | None:
    if not hex_str:
        return None
    try:
        return int(hex_str, 16)
    except (TypeError, ValueError):
        return None


def _nearest(new_phash: int, phashes: np.ndarray, image_ids: np.ndarray) -> int | None:
    if phashes.size == 0:
        return None
    distances = np.bitwise_count(phashes ^ np.uint64(new_phash))
    idx = int(np.argmin(distances))
    if int(distances[idx]) <= _PHASH_THRESHOLD:
        return int(image_ids[idx])
    return None


class Store:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def acquire_dedup_lock(self) -> None:
        await self._session.execute(select(func.pg_advisory_xact_lock(DEDUP_LOCK_KEY)))

    async def find_by_sha(self, sha256: str) -> int | None:
        return await self._session.scalar(select(Image.id).where(Image.sha256 == sha256))

    async def find_by_phash(self, phash: str | None) -> int | None:
        value = _phash_to_uint64(phash)
        if value is None:
            return None
        rows = (
            await self._session.execute(
                select(Image.id, Image.phash).where(Image.phash.is_not(None))
            )
        ).all()
        if not rows:
            return None
        ids: list[int] = []
        hashes: list[int] = []
        for image_id, stored in rows:
            parsed = _phash_to_uint64(stored)
            if parsed is None:
                continue
            ids.append(image_id)
            hashes.append(parsed)
        return _nearest(value, np.array(hashes, dtype=np.uint64), np.array(ids, dtype=np.int64))

    async def insert_canonical(
        self,
        *,
        facts: Facts,
        dataset: str | None,
        filename: str | None,
        s3_key: str,
        etag: str | None,
        file_size: int,
    ) -> int:
        image = Image(
            sha256=facts.sha256,
            dataset=dataset,
            original_filename=filename,
            s3_key=s3_key,
            s3_etag=etag,
            width=facts.width,
            height=facts.height,
            format=facts.format.lower() if facts.format else None,
            file_size=file_size,
            phash=facts.phash,
        )
        self._session.add(image)
        await self._session.flush()
        self._session.add(Processing(image_id=image.id))
        await self._session.flush()
        return image.id

    async def duplicate_view(self, image_id: int) -> ExistingImage:
        image = await self._session.get(Image, image_id)
        if image is None:
            raise ValueError(f"duplicate image {image_id} not found")
        proc = await self._session.scalar(select(Processing).where(Processing.image_id == image_id))
        if proc is None:
            proc = Processing(image_id=image_id)
            self._session.add(proc)
            await self._session.flush()
        annotation = await self._session.scalar(
            select(Annotation).where(Annotation.image_id == image_id)
        )
        needs_annotation = (
            annotation is None
            or proc.caption_status != ProcessingStatus.DONE
            or proc.ocr_status != ProcessingStatus.DONE
        )
        needs_embedding = proc.embed_status != ProcessingStatus.DONE or not proc.embed_s3_key
        return ExistingImage(
            image_id=image.id,
            s3_key=image.s3_key or "",
            sha256=image.sha256,
            width=image.width,
            height=image.height,
            format=image.format,
            needs_annotation=needs_annotation,
            needs_embedding=needs_embedding,
            existing_caption=annotation.caption_text if annotation else None,
            existing_ocr_text=annotation.ocr_text if annotation else None,
        )

    async def sweep_and_count(self, job_id: str, straggler_error: str) -> tuple[int, int, int]:
        await self._session.execute(
            update(IngestURL)
            .where(
                IngestURL.job_id == job_id,
                IngestURL.status.not_in((ProcessingStatus.DONE, ProcessingStatus.FAILED)),
            )
            .values(status=ProcessingStatus.FAILED, error_message=straggler_error)
        )
        await self._session.flush()
        processed = await self._count(
            job_id, IngestURL.status == ProcessingStatus.DONE, IngestURL.duplicate_reason.is_(None)
        )
        duplicates = await self._count(
            job_id,
            IngestURL.status == ProcessingStatus.DONE,
            IngestURL.duplicate_reason.is_not(None),
        )
        failed = await self._count(job_id, IngestURL.status == ProcessingStatus.FAILED)
        return processed, failed, duplicates

    async def _count(self, job_id: str, *conditions) -> int:
        return (
            await self._session.scalar(
                select(func.count())
                .select_from(IngestURL)
                .where(IngestURL.job_id == job_id, *conditions)
            )
            or 0
        )

    async def progress_counts(self, job_id: str) -> tuple[int, int]:
        total = (
            await self._session.scalar(
                select(func.count()).select_from(IngestURL).where(IngestURL.job_id == job_id)
            )
            or 0
        )
        completed = (
            await self._session.scalar(
                select(func.count())
                .select_from(IngestURL)
                .where(
                    IngestURL.job_id == job_id,
                    IngestURL.status.in_((ProcessingStatus.DONE, ProcessingStatus.FAILED)),
                )
            )
            or 0
        )
        return completed, total
