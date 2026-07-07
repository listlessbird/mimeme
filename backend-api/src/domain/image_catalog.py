from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import and_, false, func, or_, select
from sqlalchemy.orm import Session

from shared import db
from shared.config import settings
from shared.models import Annotation, Processing, ProcessingStatus
from shared.models import ORMImage as Image
from shared.services.storage import StorageService

ImageCatalogStatus = Literal[
    "pending",
    "downloading",
    "scanning",
    "annotating",
    "embedding",
    "done",
    "failed",
]


class ImageCatalogNotFoundError(Exception):
    """Raised when an image catalog operation targets a missing image."""


class ImageCatalogImage(BaseModel, frozen=True):
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


class ImageCatalogPage(BaseModel, frozen=True):
    images: list[ImageCatalogImage]
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


class ImageCatalog:
    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    def list_images(
        self,
        *,
        limit: int,
        offset: int,
        status: str | None = None,
        dataset: str | None = None,
        sort: Literal["newest", "oldest"] = "newest",
    ) -> ImageCatalogPage:

        with db.read_session_scope() as session:
            id_query = select(Image.id)

            if dataset:
                id_query = id_query.where(Image.dataset == dataset)

            if status:
                id_query = id_query.outerjoin(Processing, Image.id == Processing.image_id).where(
                    self._status_filter(status)
                )

            total = (
                session.execute(select(func.count()).select_from(id_query.subquery())).scalar() or 0
            )

            order_by = Image.id.desc() if sort == "newest" else Image.id.asc()
            image_ids = (
                session.execute(id_query.order_by(order_by).limit(limit).offset(offset))
                .scalars()
                .all()
            )

            images = self._load_images(session, list(image_ids))
            return ImageCatalogPage(
                images=images,
                total=total,
                limit=limit,
                offset=offset,
                has_more=(offset + len(images)) < total,
            )

    def get_image(self, image_id: int) -> ImageCatalogImage:
        with db.read_session_scope() as session:
            rows = self._load_images(session, [image_id])

        if not rows:
            raise ImageCatalogNotFoundError(f"Image {image_id} not found")
        return rows[0]

    def delete_image(self, image_id: int) -> None:

        with db.session_scope() as session:
            image = session.get(Image, image_id)

            if image is None:
                raise ImageCatalogNotFoundError(f"Image {image_id} not found")

            keys_to_delete: list[str] = []

            if image.s3_key:
                keys_to_delete.append(image.s3_key)

            if image.processing and image.processing.embed_s3_key:
                keys_to_delete.append(image.processing.embed_s3_key)
                keys_to_delete.append(image.processing.embed_s3_key.replace(".npy", "_text.npy"))

            for key in keys_to_delete:
                try:
                    self._storage.delete(key)
                except Exception:
                    pass

            session.delete(image)

    def _load_images(self, session: Session, image_ids: list[int]) -> list[ImageCatalogImage]:
        if not image_ids:
            return []

        rows = session.execute(
            select(Image, Processing, Annotation)
            .outerjoin(Processing, Processing.image_id == Image.id)
            .outerjoin(Annotation, Annotation.image_id == Image.id)
            .where(Image.id.in_(image_ids))
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
        image: Image,
        proc: Processing | None,
        ann: Annotation | None,
    ) -> ImageCatalogImage:
        url = None
        if image.s3_key:
            url = self._storage.generate_presigned_url(
                image.s3_key,
                expiration=settings.s3_presigned_url_expiry,
            )

        return ImageCatalogImage(
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
