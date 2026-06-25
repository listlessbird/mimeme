from __future__ import annotations

import datetime
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from sqlalchemy import and_, case, func, select
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


class SourceItemIngestState(StrEnum):
    INGESTED = "ingested"
    DEDUPED = "deduped"
    FAILED = "failed"
    IN_FLIGHT = "in_flight"
    UNKNOWN = "unknown"


def preview_from_metadata(raw_metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(raw_metadata, dict):
        return None

    preview = raw_metadata.get("preview")

    if isinstance(preview, list):
        urls = [url for url in preview if isinstance(url, str) and url]
        return urls[-1] if urls else None

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
    ingest_state: SourceItemIngestState
    resolved_image_id: int | None
    duplicate_reason: DuplicateReason | None
    duplicate_of_image_id: int | None
    attempt_status: ProcessingStatus | None
    attempt_error_message: str | None
    attempt_source_run_id: int | None
    media_url: str | None


class SourceItemsPage(BaseModel, frozen=True):
    items: list[SourceItemView]
    total: int
    limit: int
    offset: int
    state_counts: dict[str, int]


class RunItemView(BaseModel, frozen=True):
    id: int
    url: str
    source_item_id: int | None
    external_item_id: str | None
    title: str | None
    status: ProcessingStatus
    error_message: str | None
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

    def list_items(
        self,
        source_id: int,
        *,
        limit: int,
        offset: int,
        status: SourceItemIngestState | None = None,
    ) -> SourceItemsPage:
        self._live_source_or_raise(source_id)

        predicates = [SourceItem.source_id == source_id]
        if status is not None:
            predicates.append(self._ingest_state_predicate(status))

        total = self.db.scalar(
            select(func.count(SourceItem.id))
            .outerjoin(IngestURL, IngestURL.source_item_id == SourceItem.id)
            .where(*predicates)
        )

        rows = self.db.execute(
            select(SourceItem, IngestURL, Image.s3_key)
            .outerjoin(IngestURL, IngestURL.source_item_id == SourceItem.id)
            .outerjoin(
                Image,
                Image.id == func.coalesce(IngestURL.duplicate_of_image_id, IngestURL.image_id),
            )
            .where(*predicates)
            .order_by(SourceItem.last_seen_at.desc(), SourceItem.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()

        def presign(key: str) -> str:
            return self.storage.generate_presigned_url(
                key, expiration=settings.s3_presigned_url_expiry
            )

        items: list[SourceItemView] = []
        for item, attempt, image_s3_key in rows:
            preview_url = preview_from_metadata(item.raw_metadata)
            items.append(
                SourceItemView(
                    id=item.id,
                    external_item_id=item.external_item_id,
                    title=item.title,
                    raw_metadata=item.raw_metadata,
                    thumbnail_url=resolve_thumbnail_url(
                        image_s3_key=image_s3_key,
                        preview_url=preview_url,
                        presign=presign,
                    ),
                    first_seen_at=item.first_seen_at,
                    last_seen_at=item.last_seen_at,
                    ingest_state=self._derive_ingest_state(attempt),
                    resolved_image_id=self._resolved_image_id(attempt),
                    duplicate_reason=attempt.duplicate_reason if attempt else None,
                    duplicate_of_image_id=attempt.duplicate_of_image_id if attempt else None,
                    attempt_status=attempt.status if attempt else None,
                    attempt_error_message=attempt.error_message if attempt else None,
                    attempt_source_run_id=attempt.source_run_id if attempt else None,
                    media_url=attempt.url if attempt else None,
                )
            )

        return SourceItemsPage(
            items=items,
            total=total or 0,
            limit=limit,
            offset=offset,
            state_counts=self._state_counts(source_id),
        )

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
                    source_item_id=url.source_item_id,
                    external_item_id=source_item.external_item_id if source_item else None,
                    title=source_item.title if source_item else None,
                    status=url.status,
                    error_message=url.error_message,
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

    def _state_counts(self, source_id: int) -> dict[str, int]:
        state = case(
            (
                and_(
                    IngestURL.status == ProcessingStatus.DONE,
                    IngestURL.image_id.is_not(None),
                    IngestURL.duplicate_reason.is_(None),
                ),
                SourceItemIngestState.INGESTED.value,
            ),
            (
                and_(
                    IngestURL.status == ProcessingStatus.DONE,
                    IngestURL.duplicate_reason.is_not(None),
                    IngestURL.duplicate_of_image_id.is_not(None),
                ),
                SourceItemIngestState.DEDUPED.value,
            ),
            (IngestURL.status == ProcessingStatus.FAILED, SourceItemIngestState.FAILED.value),
            (
                IngestURL.status.in_([ProcessingStatus.PENDING, ProcessingStatus.RUNNING]),
                SourceItemIngestState.IN_FLIGHT.value,
            ),
            else_=SourceItemIngestState.UNKNOWN.value,
        )
        rows = self.db.execute(
            select(state, func.count(SourceItem.id))
            .outerjoin(IngestURL, IngestURL.source_item_id == SourceItem.id)
            .where(SourceItem.source_id == source_id)
            .group_by(state)
        ).all()
        counts = {member.value: 0 for member in SourceItemIngestState}
        for label, count in rows:
            counts[label] = count
        return counts

    def _ingest_state_predicate(self, status: SourceItemIngestState) -> Any:
        if status == SourceItemIngestState.INGESTED:
            return and_(
                IngestURL.status == ProcessingStatus.DONE,
                IngestURL.image_id.is_not(None),
                IngestURL.duplicate_reason.is_(None),
            )
        if status == SourceItemIngestState.DEDUPED:
            return and_(
                IngestURL.status == ProcessingStatus.DONE,
                IngestURL.duplicate_reason.is_not(None),
                IngestURL.duplicate_of_image_id.is_not(None),
            )
        if status == SourceItemIngestState.FAILED:
            return IngestURL.status == ProcessingStatus.FAILED
        if status == SourceItemIngestState.IN_FLIGHT:
            return IngestURL.status.in_([ProcessingStatus.PENDING, ProcessingStatus.RUNNING])
        return IngestURL.id.is_(None)

    def _derive_ingest_state(self, attempt: IngestURL | None) -> SourceItemIngestState:
        if attempt is None:
            return SourceItemIngestState.UNKNOWN
        if attempt.status in (ProcessingStatus.PENDING, ProcessingStatus.RUNNING):
            return SourceItemIngestState.IN_FLIGHT
        if attempt.status == ProcessingStatus.FAILED:
            return SourceItemIngestState.FAILED
        if (
            attempt.status == ProcessingStatus.DONE
            and attempt.duplicate_reason is not None
            and attempt.duplicate_of_image_id is not None
        ):
            return SourceItemIngestState.DEDUPED
        if (
            attempt.status == ProcessingStatus.DONE
            and attempt.image_id is not None
            and attempt.duplicate_reason is None
        ):
            return SourceItemIngestState.INGESTED
        return SourceItemIngestState.UNKNOWN

    def _resolved_image_id(self, attempt: IngestURL | None) -> int | None:
        if attempt is None:
            return None
        if attempt.duplicate_reason is not None:
            return attempt.duplicate_of_image_id
        return attempt.image_id
