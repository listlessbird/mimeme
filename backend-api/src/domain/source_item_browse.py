from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from domain.source_registry import SourceNotFoundError
from shared.config import settings
from shared.models.orm import (
    DuplicateReason,
    Image,
    IngestionSource,
    IngestURL,
    ProcessingStatus,
    SourceItem,
    SourceRun,
)


class RunNotFoundError(Exception):
    pass


def preview_from_metadata(raw_metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(raw_metadata, dict):
        return None

    preview = raw_metadata.get("preview")
    return preview if isinstance(preview, str) and preview else None


def resolve_thumbnail_url(
    *,
    image_s3_key: str | None,
    preview_url: str | None,
    presign: Callable[[str], str],
) -> str | None:
    if image_s3_key:
        return presign(image_s3_key)
    return preview_url


class SourceItemView(BaseModel, frozen=True):
    id: int
    external_item_id: str
    title: str | None
    raw_metadata: dict[str, Any] | None
    thumbnail_url: str | None
    first_seen_at: datetime.datetime
    last_seen_at: datetime.datetime


class SourceItemsPage(BaseModel, frozen=True):
    items: list[SourceItemView]
    total: int
    limit: int
    offset: int


class RunItemView(BaseModel, frozen=True):
    id: int
    url: str
    external_item_id: str | None
    title: str | None
    status: ProcessingStatus
    duplicate_reason: DuplicateReason | None
    image_id: int | None
    thumbnail_url: str | None


class RunItemsPage(BaseModel, frozen=True):
    items: list[RunItemView]
    total: int
    limit: int
    offset: int


class SourceItemBrowser:
    def __init__(self, db: Session, storage: Any) -> None:
        self.db = db
        self.storage = storage

    def list_items(self, source_id: int, *, limit: int, offset: int) -> SourceItemsPage:
        self._live_source_or_raise(source_id)

        total = self.db.scalar(
            select(func.count(SourceItem.id)).where(SourceItem.source_id == source_id)
        )

        rows = (
            self.db.execute(
                select(SourceItem)
                .where(SourceItem.source_id == source_id)
                .order_by(SourceItem.last_seen_at.desc(), SourceItem.id.desc())
                .limit(limit)
                .offset(offset)
            )
            .scalars()
            .all()
        )

        items = [
            SourceItemView(
                id=item.id,
                external_item_id=item.external_item_id,
                title=item.title,
                raw_metadata=item.raw_metadata,
                thumbnail_url=preview_from_metadata(item.raw_metadata),
                first_seen_at=item.first_seen_at,
                last_seen_at=item.last_seen_at,
            )
            for item in rows
        ]

        return SourceItemsPage(items=items, total=total or 0, limit=limit, offset=offset)

    def list_run_items(
        self, source_id: int, run_id: int, *, limit: int, offset: int
    ) -> RunItemsPage:
        self._live_source_or_raise(source_id)

        run = self.db.execute(
            select(SourceRun.id).where(SourceRun.id == run_id, SourceRun.source_id == source_id)
        ).scalar_one_or_none()
        if run is None:
            raise RunNotFoundError(run_id)

        total = self.db.scalar(
            select(func.count(IngestURL.id)).where(IngestURL.source_run_id == run_id)
        )

        rows = (
            self.db.execute(
                select(IngestURL)
                .where(IngestURL.source_run_id == run_id)
                .order_by(IngestURL.id.asc())
                .limit(limit)
                .offset(offset)
            )
            .scalars()
            .all()
        )

        s3_key_by_image_id = self._s3_keys_for(
            {
                resolved
                for url in rows
                if (resolved := url.image_id or url.duplicate_of_image_id) is not None
            }
        )
        item_by_id = self._source_items_for(
            {url.source_item_id for url in rows if url.source_item_id is not None}
        )

        def presign(key: str) -> str:
            return self.storage.generate_presigned_url(
                key, expiration=settings.s3_presigned_url_expiry
            )

        items: list[RunItemView] = []
        for url in rows:
            resolved_image_id = url.image_id or url.duplicate_of_image_id
            source_item = (
                item_by_id.get(url.source_item_id) if url.source_item_id is not None else None
            )
            items.append(
                RunItemView(
                    id=url.id,
                    url=url.url,
                    external_item_id=source_item.external_item_id if source_item else None,
                    title=source_item.title if source_item else None,
                    status=url.status,
                    duplicate_reason=url.duplicate_reason,
                    image_id=resolved_image_id,
                    thumbnail_url=resolve_thumbnail_url(
                        image_s3_key=(
                            s3_key_by_image_id.get(resolved_image_id)
                            if resolved_image_id is not None
                            else None
                        ),
                        preview_url=preview_from_metadata(
                            source_item.raw_metadata if source_item else None
                        ),
                        presign=presign,
                    ),
                )
            )

        return RunItemsPage(items=items, total=total or 0, limit=limit, offset=offset)

    def _live_source_or_raise(self, source_id: int) -> None:
        exists = self.db.execute(
            select(IngestionSource.id).where(
                IngestionSource.id == source_id, IngestionSource.deleted_at.is_(None)
            )
        ).scalar_one_or_none()
        if exists is None:
            raise SourceNotFoundError(source_id)

    def _s3_keys_for(self, image_ids: set[int]) -> dict[int, str | None]:
        if not image_ids:
            return {}
        rows = self.db.execute(select(Image.id, Image.s3_key).where(Image.id.in_(image_ids))).all()
        return {image_id: s3_key for image_id, s3_key in rows}

    def _source_items_for(self, item_ids: set[int]) -> dict[int, SourceItem]:
        if not item_ids:
            return {}
        rows = (
            self.db.execute(select(SourceItem).where(SourceItem.id.in_(item_ids))).scalars().all()
        )
        return {item.id: item for item in rows}
