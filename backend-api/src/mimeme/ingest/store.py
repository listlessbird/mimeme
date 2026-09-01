from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mimeme.db.schema import (
    Annotation,
    Image,
    IngestionSource,
    IngestURL,
    Processing,
    ProcessingStatus,
    SourceItem,
)
from mimeme.inference.model import Context
from mimeme.ingest.facts import Facts
from mimeme.ingest.model import Result
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
    caption_context_sha256: str | None
    caption_prompt_version: str | None


class PHashMatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: int
    distance: int


def _phash_to_uint64(hex_str: str | None) -> int | None:
    if not hex_str:
        return None
    try:
        return int(hex_str, 16)
    except (TypeError, ValueError):
        return None


def _nearest(new_phash: int, candidates: list[tuple[int, int]]) -> PHashMatch | None:
    nearest_id: int | None = None
    nearest_distance = _PHASH_THRESHOLD + 1
    for image_id, stored_phash in candidates:
        distance = (new_phash ^ stored_phash).bit_count()
        if distance < nearest_distance:
            nearest_id = image_id
            nearest_distance = distance
    if nearest_id is None or nearest_distance > _PHASH_THRESHOLD:
        return None
    return PHashMatch(image_id=nearest_id, distance=nearest_distance)


class Store:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def acquire_dedup_lock(self) -> None:
        await self._session.execute(select(func.pg_advisory_xact_lock(DEDUP_LOCK_KEY)))

    async def terminal_result(self, ingest_url_id: int) -> Result | None:
        item = await self._session.get(IngestURL, ingest_url_id)
        if item is None or item.status not in (ProcessingStatus.DONE, ProcessingStatus.FAILED):
            return None
        if item.status == ProcessingStatus.FAILED:
            return Result(item_id=item.id, outcome="failed", error=item.error_message)
        return Result(
            item_id=item.id,
            outcome="duplicate" if item.duplicate_reason is not None else "processed",
            image_id=item.image_id,
            duplicate_reason=item.duplicate_reason,
        )

    async def inference_context(self, ingest_url_id: int) -> Context | None:
        raw = await self._session.scalar(
            select(SourceItem.known_facts)
            .join(IngestURL, IngestURL.source_item_id == SourceItem.id)
            .where(IngestURL.id == ingest_url_id)
        )
        if not raw:
            return None
        context = Context.model_validate(raw)
        return context if any(context.model_dump().values()) else None

    async def phash_dedup_allowed(self, ingest_url_id: int) -> bool:
        adapter_key = await self._session.scalar(
            select(IngestionSource.adapter_key)
            .join(IngestURL, IngestURL.source_id == IngestionSource.id)
            .where(IngestURL.id == ingest_url_id)
        )
        return adapter_key != "kym"

    async def find_by_sha(self, sha256: str) -> int | None:
        return await self._session.scalar(select(Image.id).where(Image.sha256 == sha256))

    async def find_by_phash(self, phash: str | None) -> int | None:
        match = await self.find_phash_match(phash)
        return match.image_id if match is not None else None

    async def find_phash_match(self, phash: str | None) -> PHashMatch | None:
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
        candidates: list[tuple[int, int]] = []
        for image_id, stored in rows:
            parsed = _phash_to_uint64(stored)
            if parsed is None:
                continue
            candidates.append((image_id, parsed))
        return _nearest(value, candidates)

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
            caption_context_sha256=(annotation.caption_context_sha256 if annotation else None),
            caption_prompt_version=(annotation.caption_prompt_version if annotation else None),
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
