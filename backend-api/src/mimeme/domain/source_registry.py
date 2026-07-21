import datetime
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from mimeme.db.schema import (
    IngestionSource,
    IngestURL,
    ProcessingStatus,
    SourceItem,
    SourceRun,
    SourceRunStatus,
    SourceRunTrigger,
)
from mimeme.domain.adapters.registry import KNOWN_ADAPTER_KEYS, UnknownAdapterKeyError
from mimeme.domain.source_run_accounting import UrlOutcome, derive_run_accounting
from mimeme.domain.source_schedule_spec import ScheduleSpec, derive_schedule_spec
from mimeme.shared import db


class SourceNotFoundError(Exception):
    pass


class DuplicateSourceNameError(Exception):
    pass


class _Unset:
    pass


UNSET = _Unset()


class SourceView(BaseModel, frozen=True):
    id: int
    name: str
    adapter_key: str
    adapter_config: dict[str, Any]
    dataset: str | None
    schedule_cron: str | None
    schedule_timezone: str
    max_items_per_run: int | None
    enabled: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime


class SourceStats(BaseModel, frozen=True):
    run_count: int
    items_discovered: int
    duplicate_count: int
    images_ingested: int
    failed_count: int


class SourceRunView(BaseModel, frozen=True):
    id: int
    trigger_mode: SourceRunTrigger
    status: SourceRunStatus
    ingest_job_id: str | None
    error_message: str | None
    started_at: datetime.datetime | None
    completed_at: datetime.datetime | None
    created_at: datetime.datetime
    discovered: int
    queued: int
    duplicate: int
    failed: int


class SourceListItem(SourceView, frozen=True):
    stats: SourceStats


class SourceDetail(SourceView, frozen=True):
    stats: SourceStats
    recent_runs: list[SourceRunView]


class SourceRegistry:
    async def create(
        self,
        *,
        name: str,
        adapter_key: str,
        adapter_config: dict[str, Any],
        dataset: str | None,
        schedule_cron: str | None,
        schedule_timezone: str,
        max_items_per_run: int | None,
        enabled: bool = True,
    ) -> SourceView:

        if adapter_key not in KNOWN_ADAPTER_KEYS:
            raise UnknownAdapterKeyError(adapter_key)

        async with db.write_session() as session:
            if await self._live_name_exists(session, name):
                raise DuplicateSourceNameError(name)

            source = IngestionSource(
                name=name,
                adapter_key=adapter_key,
                adapter_config=adapter_config,
                dataset=dataset,
                schedule_cron=schedule_cron,
                schedule_timezone=schedule_timezone,
                max_items_per_run=max_items_per_run,
                enabled=enabled,
            )

            session.add(source)
            await session.flush()
            await session.refresh(source)

            return self._to_source_view(source)

    async def list_sources(self) -> list[SourceListItem]:

        async with db.read_session() as session:
            sources = (
                await session.scalars(
                    select(IngestionSource)
                    .where(IngestionSource.deleted_at.is_(None))
                    .order_by(IngestionSource.created_at.desc())
                )
            ).all()

            stats_by_souce_id = await self._stats_by_source_id(
                session, [source.id for source in sources]
            )

            return [
                SourceListItem(
                    **self._to_source_view(source).model_dump(),
                    stats=stats_by_souce_id[source.id],
                )
                for source in sources
            ]

    async def get_source(self, source_id: int, *, recent_runs_limit: int = 20) -> SourceDetail:

        async with db.read_session() as session:
            source = await self._live_source_or_raise(session, source_id)

            runs = (
                await session.scalars(
                    select(SourceRun)
                    .where(SourceRun.source_id == source.id)
                    .order_by(SourceRun.created_at.desc())
                    .limit(recent_runs_limit)
                )
            ).all()

            return SourceDetail(
                **self._to_source_view(source).model_dump(),
                stats=(await self._stats_by_source_id(session, [source.id]))[source.id],
                recent_runs=await self._to_run_views(session, runs),
            )

    async def patch(
        self,
        source_id: int,
        *,
        adapter_config: dict[str, Any] | _Unset = UNSET,
        dataset: str | None | _Unset = UNSET,
        schedule_cron: str | None | _Unset = UNSET,
        schedule_timezone: str | _Unset = UNSET,
        max_items_per_run: int | None | _Unset = UNSET,
        enabled: bool | _Unset = UNSET,
    ) -> SourceView:

        async with db.write_session() as session:
            source = await self._live_source_or_raise(session, source_id)

            if not isinstance(adapter_config, _Unset):
                source.adapter_config = adapter_config

            if not isinstance(dataset, _Unset):
                source.dataset = dataset

            if not isinstance(schedule_cron, _Unset):
                source.schedule_cron = schedule_cron

            if not isinstance(schedule_timezone, _Unset):
                source.schedule_timezone = schedule_timezone

            if not isinstance(max_items_per_run, _Unset):
                source.max_items_per_run = max_items_per_run

            if not isinstance(enabled, _Unset):
                source.enabled = enabled

            await session.flush()
            await session.refresh(source)

            return self._to_source_view(source)

    async def soft_delete(self, source_id: int) -> None:

        async with db.write_session() as session:
            source = await self._live_source_or_raise(session, source_id)

            source.deleted_at = datetime.datetime.now(datetime.UTC)
            await session.flush()

    async def list_schedule_specs(self) -> list[ScheduleSpec]:
        async with db.read_session() as session:
            sources = (
                await session.scalars(
                    select(IngestionSource).where(IngestionSource.deleted_at.is_(None))
                )
            ).all()

            return [
                derive_schedule_spec(
                    source_id=source.id,
                    schedule_cron=source.schedule_cron,
                    schedule_timezone=source.schedule_timezone,
                    enabled=source.enabled,
                    deleted=False,
                )
                for source in sources
            ]

    async def _live_name_exists(self, session: AsyncSession, name: str) -> bool:

        return (
            await session.scalars(
                select(IngestionSource).where(
                    IngestionSource.name == name, IngestionSource.deleted_at.is_(None)
                )
            )
        ).first() is not None

    async def _live_source_or_raise(self, session: AsyncSession, source_id: int) -> IngestionSource:
        source = (
            await session.scalars(
                select(IngestionSource).where(
                    IngestionSource.id == source_id, IngestionSource.deleted_at.is_(None)
                )
            )
        ).one_or_none()

        if source is None:
            raise SourceNotFoundError(source_id)

        return source

    async def _stats_by_source_id(
        self, session: AsyncSession, source_ids: list[int]
    ) -> dict[int, SourceStats]:
        stats = {
            source_id: {
                "run_count": 0,
                "items_discovered": 0,
                "duplicate_count": 0,
                "images_ingested": 0,
                "failed_count": 0,
            }
            for source_id in source_ids
        }

        if not source_ids:
            return {}

        run_rows = (
            await session.execute(
                select(SourceRun.source_id, func.count(SourceRun.id))
                .where(SourceRun.source_id.in_(source_ids))
                .group_by(SourceRun.source_id)
            )
        ).all()

        item_rows = (
            await session.execute(
                select(SourceItem.source_id, func.count(SourceItem.id))
                .where(SourceItem.source_id.in_(source_ids))
                .group_by(SourceItem.source_id)
            )
        ).all()

        url_rows = (
            await session.execute(
                select(
                    IngestURL.source_id,
                    func.count(IngestURL.id).filter(IngestURL.duplicate_reason.is_not(None)),
                    func.count(IngestURL.id).filter(
                        IngestURL.status == ProcessingStatus.DONE,
                        IngestURL.image_id.is_not(None),
                        IngestURL.duplicate_reason.is_(None),
                    ),
                    func.count(IngestURL.id).filter(IngestURL.status == ProcessingStatus.FAILED),
                )
                .where(IngestURL.source_id.in_(source_ids))
                .group_by(IngestURL.source_id)
            )
        ).all()

        for source_id, count in run_rows:
            stats[source_id]["run_count"] = count

        for source_id, count in item_rows:
            stats[source_id]["items_discovered"] = count

        for source_id, duplicate_count, images_ingested, failed_count in url_rows:
            stats[source_id]["duplicate_count"] = duplicate_count
            stats[source_id]["images_ingested"] = images_ingested
            stats[source_id]["failed_count"] = failed_count

        return {source_id: SourceStats(**values) for source_id, values in stats.items()}

    async def _to_run_views(
        self, session: AsyncSession, runs: Sequence[SourceRun]
    ) -> list[SourceRunView]:
        run_ids = [run.id for run in runs]

        if not run_ids:
            return []

        discovered_by_run_id = await self._discovered_count_by_run_id(session, run_ids)
        outcomes_by_run_id = await self._url_outcomes_by_run_id(session, run_ids)

        views: list[SourceRunView] = []

        for run in runs:
            accounting = derive_run_accounting(
                discovered_items=discovered_by_run_id.get(run.id, 0),
                url_outcomes=outcomes_by_run_id.get(run.id, []),
            )

            views.append(
                SourceRunView(
                    id=run.id,
                    trigger_mode=run.trigger_mode,
                    status=run.status,  # stored status, do not use accounting.status
                    ingest_job_id=run.ingest_job_id,
                    error_message=run.error_message,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                    created_at=run.created_at,
                    discovered=accounting.discovered,
                    queued=accounting.queued,
                    duplicate=accounting.duplicate,
                    failed=accounting.failed,
                )
            )

        return views

    async def _discovered_count_by_run_id(
        self, session: AsyncSession, run_ids: list[int]
    ) -> dict[int, int]:
        rows = (
            await session.execute(
                select(SourceItem.last_source_run_id, func.count(SourceItem.id))
                .where(SourceItem.last_source_run_id.in_(run_ids))
                .group_by(SourceItem.last_source_run_id)
            )
        ).all()

        return {run_id: count for run_id, count in rows if run_id is not None}

    async def _url_outcomes_by_run_id(
        self, session: AsyncSession, run_ids: list[int]
    ) -> dict[int, list[UrlOutcome]]:

        rows = (
            await session.execute(
                select(IngestURL.source_run_id, IngestURL.status, IngestURL.duplicate_reason).where(
                    IngestURL.source_run_id.in_(run_ids)
                )
            )
        ).all()

        outcomes_by_run_id: dict[int, list[UrlOutcome]] = {run_id: [] for run_id in run_ids}

        for run_id, status, duplicate_reason in rows:
            if run_id is None:
                continue

            outcomes_by_run_id[run_id].append(
                UrlOutcome(
                    status=status,
                    duplicate_reason=duplicate_reason,
                )
            )

        return outcomes_by_run_id

    def _to_source_view(self, source: IngestionSource) -> SourceView:
        return SourceView(
            id=source.id,
            name=source.name,
            adapter_key=source.adapter_key,
            adapter_config=source.adapter_config,
            dataset=source.dataset,
            schedule_cron=source.schedule_cron,
            schedule_timezone=source.schedule_timezone,
            max_items_per_run=source.max_items_per_run,
            enabled=source.enabled,
            created_at=source.created_at,
            updated_at=source.updated_at,
        )
