from __future__ import annotations

import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from sqlalchemy import ColumnElement, and_, func, or_, select

from mimeme.db import Db
from mimeme.db.schema import (
    DuplicateReason,
    Image,
    IngestionSource,
    IngestStage,
    IngestURL,
    ProcessingStatus,
    SourceItem,
    SourceRun,
    SourceRunTrigger,
)
from mimeme.ingest.model import Source, restore
from mimeme.media import Urls

DEFAULT_LIVE_WINDOW = datetime.timedelta(minutes=5)


class NotFound(Exception):
    pass


class View(StrEnum):
    LIVE = "live"
    COMPLETED = "completed"
    FAILED = "failed"
    ALL = "all"


class Outcome(StrEnum):
    INGESTED = "ingested"
    DEDUPED = "deduped"
    FAILED = "failed"
    IN_FLIGHT = "in_flight"


class Row(BaseModel, frozen=True):
    ingest_url_id: int
    input: Source
    job_id: str
    source_run_id: int | None
    source_id: int | None
    source_name: str | None
    trigger: SourceRunTrigger
    stage: IngestStage
    status: ProcessingStatus
    outcome: Outcome
    duplicate_reason: DuplicateReason | None
    duplicate_of_image_id: int | None
    resolved_image_id: int | None
    dataset: str | None
    thumbnail_url: str | None
    error_message: str | None
    created_at: datetime.datetime
    stage_updated_at: datetime.datetime | None


class Page(BaseModel, frozen=True):
    rows: list[Row]
    total: int
    limit: int
    offset: int


class Detail(BaseModel, frozen=True):
    ingest_url_id: int
    input: Source
    job_id: str
    source_run_id: int | None
    source_id: int | None
    source_name: str | None
    trigger: SourceRunTrigger
    stage: IngestStage
    status: ProcessingStatus
    outcome: Outcome
    duplicate_reason: DuplicateReason | None
    duplicate_of_image_id: int | None
    resolved_image_id: int | None
    image_id: int | None
    dataset: str | None
    thumbnail_url: str | None
    error_message: str | None
    created_at: datetime.datetime
    stage_updated_at: datetime.datetime | None


_RESOLVED_IMAGE_ID = func.coalesce(IngestURL.duplicate_of_image_id, IngestURL.image_id)

_INGESTED = and_(
    IngestURL.status == ProcessingStatus.DONE,
    IngestURL.image_id.is_not(None),
    IngestURL.duplicate_reason.is_(None),
)
_DEDUPED = and_(
    IngestURL.status == ProcessingStatus.DONE,
    IngestURL.duplicate_reason.is_not(None),
    IngestURL.duplicate_of_image_id.is_not(None),
)


class Browse:
    def __init__(self, db: Db, media_urls: Urls) -> None:
        self._db = db
        self.media_urls = media_urls

    async def list_attempts(
        self,
        *,
        limit: int,
        offset: int,
        view: View = View.ALL,
        stage: IngestStage | None = None,
        trigger: SourceRunTrigger | None = None,
        source_id: int | None = None,
        dataset: str | None = None,
        outcome: Outcome | None = None,
        created_from: datetime.datetime | None = None,
        created_to: datetime.datetime | None = None,
        live_window: datetime.timedelta = DEFAULT_LIVE_WINDOW,
        now: datetime.datetime | None = None,
    ) -> Page:

        async with self._db.read_session() as session:
            predicates = self._predicates(
                view=view,
                stage=stage,
                trigger=trigger,
                source_id=source_id,
                dataset=dataset,
                outcome=outcome,
                created_from=created_from,
                created_to=created_to,
                live_window=live_window,
                now=now,
            )

            total = await session.scalar(
                self._with_joins(select(func.count(IngestURL.id)).select_from(IngestURL)).where(
                    *predicates
                )
            )

            rows = (
                await session.execute(
                    self._with_joins(
                        select(
                            IngestURL,
                            IngestionSource.name,
                            SourceRun.trigger_mode,
                            Image.s3_key,
                            Image.dataset,
                            IngestionSource.dataset,
                            SourceItem.raw_metadata,
                        ).select_from(IngestURL)
                    )
                    .where(*predicates)
                    .order_by(IngestURL.created_at.desc(), IngestURL.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
            ).all()

            return Page(
                rows=[self._row(*r) for r in rows],
                total=total or 0,
                limit=limit,
                offset=offset,
            )

    async def get_attempt(self, ingest_url_id: int) -> Detail:

        async with self._db.read_session() as session:
            row = (
                await session.execute(
                    self._with_joins(
                        select(
                            IngestURL,
                            IngestionSource.name,
                            SourceRun.trigger_mode,
                            Image.s3_key,
                            Image.dataset,
                            IngestionSource.dataset,
                            SourceItem.raw_metadata,
                        ).select_from(IngestURL)
                    ).where(IngestURL.id == ingest_url_id)
                )
            ).first()

        if row is None:
            raise NotFound(ingest_url_id)

        attempt = row[0]
        base = self._row(*row)
        return Detail(
            **base.model_dump(),
            image_id=attempt.image_id,
        )

    def _with_joins(self, stmt: Any) -> Any:
        return (
            stmt.outerjoin(SourceRun, SourceRun.id == IngestURL.source_run_id)
            .outerjoin(IngestionSource, IngestionSource.id == IngestURL.source_id)
            .outerjoin(Image, Image.id == _RESOLVED_IMAGE_ID)
            .outerjoin(SourceItem, SourceItem.id == IngestURL.source_item_id)
        )

    def _predicates(
        self,
        *,
        view: View,
        stage: IngestStage | None,
        trigger: SourceRunTrigger | None,
        source_id: int | None,
        dataset: str | None,
        outcome: Outcome | None,
        created_from: datetime.datetime | None,
        created_to: datetime.datetime | None,
        live_window: datetime.timedelta,
        now: datetime.datetime | None,
    ) -> list[ColumnElement[bool]]:
        predicates: list[ColumnElement[bool]] = []

        view_predicate = self._view_predicate(view, live_window=live_window, now=now)
        if view_predicate is not None:
            predicates.append(view_predicate)

        if stage is not None:
            predicates.append(IngestURL.stage == stage)

        if trigger is not None:
            predicates.append(self._trigger_predicate(trigger))

        if source_id is not None:
            predicates.append(IngestURL.source_id == source_id)

        if dataset is not None:
            predicates.append(func.coalesce(Image.dataset, IngestionSource.dataset) == dataset)

        if outcome is not None:
            predicates.append(self._outcome_predicate(outcome))

        if created_from is not None:
            predicates.append(IngestURL.created_at >= created_from)
        if created_to is not None:
            predicates.append(IngestURL.created_at <= created_to)

        return predicates

    def _view_predicate(
        self,
        view: View,
        *,
        live_window: datetime.timedelta,
        now: datetime.datetime | None,
    ) -> ColumnElement[bool] | None:
        if view == View.ALL:
            return None
        if view == View.FAILED:
            return IngestURL.status == ProcessingStatus.FAILED
        if view == View.COMPLETED:
            return _INGESTED | _DEDUPED
        # LIVE: in-flight, plus terminal attempts finished within the window.
        reference = now or datetime.datetime.now(datetime.UTC)
        cutoff = reference - live_window
        in_flight = IngestURL.status.in_([ProcessingStatus.PENDING, ProcessingStatus.RUNNING])
        recently_finished = and_(
            IngestURL.status.in_([ProcessingStatus.DONE, ProcessingStatus.FAILED]),
            func.coalesce(IngestURL.stage_updated_at, IngestURL.created_at) >= cutoff,
        )
        return or_(in_flight, recently_finished)

    def _trigger_predicate(self, trigger: SourceRunTrigger) -> ColumnElement[bool]:
        if trigger == SourceRunTrigger.SCHEDULED:
            return SourceRun.trigger_mode == SourceRunTrigger.SCHEDULED
        return or_(
            IngestURL.source_run_id.is_(None),
            SourceRun.trigger_mode == SourceRunTrigger.MANUAL,
        )

    def _outcome_predicate(self, outcome: Outcome) -> ColumnElement[bool]:
        if outcome == Outcome.INGESTED:
            return _INGESTED
        if outcome == Outcome.DEDUPED:
            return _DEDUPED
        if outcome == Outcome.FAILED:
            return IngestURL.status == ProcessingStatus.FAILED
        return IngestURL.status.in_([ProcessingStatus.PENDING, ProcessingStatus.RUNNING])

    def _row(
        self,
        attempt: IngestURL,
        source_name: str | None,
        trigger_mode: SourceRunTrigger | None,
        image_s3_key: str | None,
        image_dataset: str | None,
        source_dataset: str | None,
        raw_metadata: dict[str, Any] | None,
    ) -> Row:
        return Row(
            ingest_url_id=attempt.id,
            input=restore(
                kind=attempt.input_kind,
                url=attempt.url,
                artifact_key=attempt.artifact_key,
            ),
            job_id=attempt.job_id,
            source_run_id=attempt.source_run_id,
            source_id=attempt.source_id,
            source_name=source_name,
            trigger=self._derive_trigger(attempt, trigger_mode),
            stage=attempt.stage,
            status=attempt.status,
            outcome=self._derive_outcome(attempt),
            duplicate_reason=attempt.duplicate_reason,
            duplicate_of_image_id=attempt.duplicate_of_image_id,
            resolved_image_id=self._resolved_image_id(attempt),
            dataset=image_dataset or source_dataset,
            thumbnail_url=_thumbnail(
                image_s3_key=image_s3_key,
                preview_url=_preview(raw_metadata),
                media_urls=self.media_urls,
            ),
            error_message=attempt.error_message,
            created_at=attempt.created_at,
            stage_updated_at=attempt.stage_updated_at,
        )

    def _derive_trigger(
        self, attempt: IngestURL, trigger_mode: SourceRunTrigger | None
    ) -> SourceRunTrigger:
        if attempt.source_run_id is not None and trigger_mode is not None:
            return trigger_mode
        return SourceRunTrigger.MANUAL

    def _derive_outcome(self, attempt: IngestURL) -> Outcome:
        if attempt.status == ProcessingStatus.FAILED:
            return Outcome.FAILED
        if attempt.status == ProcessingStatus.DONE:
            if attempt.duplicate_reason is not None and attempt.duplicate_of_image_id is not None:
                return Outcome.DEDUPED
            return Outcome.INGESTED
        return Outcome.IN_FLIGHT

    def _resolved_image_id(self, attempt: IngestURL) -> int | None:
        if attempt.duplicate_reason is not None:
            return attempt.duplicate_of_image_id
        return attempt.image_id


def _preview(raw_metadata: dict[str, Any] | None) -> str | None:
    if not raw_metadata:
        return None
    value = raw_metadata.get("preview")
    if isinstance(value, list):
        urls = [item for item in value if isinstance(item, str) and item]
        return urls[-1] if urls else None
    return value if isinstance(value, str) and value else None


def _thumbnail(
    *, image_s3_key: str | None, preview_url: str | None, media_urls: Urls
) -> str | None:
    return media_urls.resolve(image_s3_key) if image_s3_key else preview_url
