from __future__ import annotations

from datetime import datetime
from typing import Literal

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import and_, false, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from mimeme import storage
from mimeme.db import Db
from mimeme.db.schema import Annotation, Processing, ProcessingStatus
from mimeme.db.schema import ORMImage as ImageRow
from mimeme.job.store import Store as JobStore
from mimeme.media import Urls

ImageCatalogStatus = Literal[
    "pending",
    "downloading",
    "scanning",
    "annotating",
    "embedding",
    "done",
    "failed",
]
log = structlog.get_logger()


class NotFound(Exception):
    """Raised when an image catalog operation targets a missing image."""


class Image(BaseModel, frozen=True):
    id: int
    sha256: str
    status: ImageCatalogStatus
    url: str | None = None
    s3_key: str | None = None
    dataset: str | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None
    phash: str | None = None
    file_size: int | None = None
    ocr_status: str | None = None
    caption_status: str | None = None
    embed_status: str | None = None
    ocr_model: str | None = None
    caption_model: str | None = None
    embed_model: str | None = None
    embed_dim: int | None = None
    embed_s3_key: str | None = None
    caption: str | None = None
    ocr_text: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class Page(BaseModel, frozen=True):
    images: list[Image]
    total: int
    limit: int
    offset: int
    has_more: bool


def project_image_status(proc: Processing | None) -> ImageCatalogStatus:
    if not proc:
        return "pending"

    if proc.embed_status == ProcessingStatus.DONE:
        return "done"

    if any(
        status == ProcessingStatus.FAILED
        for status in [proc.ocr_status, proc.caption_status, proc.embed_status]
    ):
        return "failed"

    if proc.embed_status == ProcessingStatus.RUNNING:
        return "embedding"

    if (
        proc.caption_status == ProcessingStatus.RUNNING
        or proc.ocr_status == ProcessingStatus.RUNNING
    ):
        return "annotating"

    return "pending"


class Catalog:
    def __init__(
        self,
        db: Db,
        media_storage: storage.Store,
        artifact_storage: storage.Store,
        media_urls: Urls,
    ) -> None:
        self._db = db
        self._media_storage = media_storage
        self._artifact_storage = artifact_storage
        self._media_urls = media_urls

    async def list_images(
        self,
        *,
        limit: int,
        offset: int,
        status: str | None = None,
        dataset: str | None = None,
        sort: Literal["newest", "oldest"] = "newest",
    ) -> Page:

        async with self._db.read_session() as session:
            id_query = select(ImageRow.id)

            if dataset:
                id_query = id_query.where(ImageRow.dataset == dataset)

            if status:
                id_query = id_query.outerjoin(Processing, ImageRow.id == Processing.image_id).where(
                    self._status_filter(status)
                )

            row = await session.execute(select(func.count()).select_from(id_query.subquery()))

            total = row.scalar() or 0

            order_by = ImageRow.id.desc() if sort == "newest" else ImageRow.id.asc()

            image_id_rows = await session.execute(
                id_query.order_by(order_by).limit(limit).offset(offset)
            )

            image_ids = image_id_rows.scalars().all()

            images = await self._load_images(session, list(image_ids))
            return Page(
                images=images,
                total=total,
                limit=limit,
                offset=offset,
                has_more=(offset + len(images)) < total,
            )

    async def get_image(self, image_id: int) -> Image:
        async with self._db.read_session() as session:
            rows = await self._load_images(session, [image_id])

        if not rows:
            raise NotFound(f"Image {image_id} not found")
        return rows[0]

    async def delete_image(self, image_id: int) -> None:

        async with self._db.read_session() as session:
            row = (
                await session.execute(
                    select(ImageRow, Processing)
                    .outerjoin(Processing, Processing.image_id == ImageRow.id)
                    .where(ImageRow.id == image_id)
                )
            ).one_or_none()

            if row is None:
                raise NotFound()

            image, processing = row
            media_key = image.s3_key
            embedding_key = processing.embed_s3_key if processing else None
            was_searchable = (
                processing is not None
                and processing.embed_status == ProcessingStatus.DONE
                and processing.embed_s3_key is not None
            )

        if media_key:
            await self._delete(self._media_storage, media_key, role="media")
        if embedding_key:
            await self._delete(self._artifact_storage, embedding_key, role="artifact")

        async with self._db.write_session() as session:
            image = await session.get(ImageRow, image_id)
            if image is None:
                raise NotFound()
            await session.delete(image)
            if was_searchable:
                await JobStore(session).mark_dirty(reason="image_deleted")

    async def _delete(self, store: storage.Store, key: str, *, role: str) -> None:
        try:
            await store.delete(storage.Object(key))
        except storage.Error:
            log.warning(
                "image_catalog_storage_delete_failed",
                storage_role=role,
                key=key,
                exc_info=True,
            )

    async def _load_images(self, session: AsyncSession, image_ids: list[int]) -> list[Image]:
        if not image_ids:
            return []

        rows = (
            await session.execute(
                select(ImageRow, Processing, Annotation)
                .outerjoin(Processing, Processing.image_id == ImageRow.id)
                .outerjoin(Annotation, Annotation.image_id == ImageRow.id)
                .where(ImageRow.id.in_(image_ids))
            )
        ).all()

        row_by_id = {image.id: (image, proc, ann) for image, proc, ann in rows}

        return [
            self._project_image(image, proc, ann)
            for image_id in image_ids
            if (row := row_by_id.get(image_id))
            for image, proc, ann in [row]
        ]

    def _project_image(
        self,
        image: ImageRow,
        proc: Processing | None,
        ann: Annotation | None,
    ) -> Image:
        url = None
        if image.s3_key:
            url = self._media_urls.resolve(image.s3_key)

        return Image(
            id=image.id,
            sha256=image.sha256,
            url=url,
            s3_key=image.s3_key,
            dataset=image.dataset,
            width=image.width,
            height=image.height,
            format=image.format,
            phash=image.phash,
            file_size=image.file_size,
            status=project_image_status(proc),
            ocr_status=proc.ocr_status.value if proc else None,
            caption_status=proc.caption_status.value if proc else None,
            embed_status=proc.embed_status.value if proc else None,
            ocr_model=proc.ocr_model if proc else None,
            caption_model=proc.caption_model if proc else None,
            embed_model=proc.embed_model if proc else None,
            embed_dim=proc.embed_dim if proc else None,
            embed_s3_key=proc.embed_s3_key if proc else None,
            caption=ann.caption_text if ann else None,
            ocr_text=ann.ocr_text if ann else None,
            tags=[],
            created_at=image.created_at,
        )

    def _status_filter(self, status: str):
        no_failures = and_(
            Processing.ocr_status != ProcessingStatus.FAILED,
            Processing.caption_status != ProcessingStatus.FAILED,
            Processing.embed_status != ProcessingStatus.FAILED,
        )
        not_done = Processing.embed_status != ProcessingStatus.DONE

        if status == "done":
            return Processing.embed_status == ProcessingStatus.DONE
        if status == "failed":
            return and_(
                not_done,
                or_(
                    Processing.ocr_status == ProcessingStatus.FAILED,
                    Processing.caption_status == ProcessingStatus.FAILED,
                    Processing.embed_status == ProcessingStatus.FAILED,
                ),
            )
        if status == "embedding":
            return and_(not_done, no_failures, Processing.embed_status == ProcessingStatus.RUNNING)
        if status == "annotating":
            return and_(
                not_done,
                no_failures,
                Processing.embed_status != ProcessingStatus.RUNNING,
                or_(
                    Processing.caption_status == ProcessingStatus.RUNNING,
                    Processing.ocr_status == ProcessingStatus.RUNNING,
                ),
            )
        if status == "pending":
            return or_(
                Processing.image_id.is_(None),
                and_(
                    not_done,
                    no_failures,
                    Processing.embed_status != ProcessingStatus.RUNNING,
                    Processing.caption_status != ProcessingStatus.RUNNING,
                    Processing.ocr_status != ProcessingStatus.RUNNING,
                ),
            )
        return false()
