import datetime
from typing import Any, Final

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from domain.source_run_accounting import UrlOutcome, derive_run_accounting
from shared.models.orm import (
    IngestionSource,
    IngestURL,
    SourceItem,
    SourceRun,
    SourceRunStatus,
    SourceRunTrigger,
)

KNOWN_ADAPTERS: Final = frozenset({"meme_api"})


class SourceNotFoundError(Exception):
    pass


class UnknownAdapterKeyError(Exception):
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


class SourceRunView(BaseModel, frozen=True):
    id: int
    trigger_mode: SourceRunTrigger
    status: SourceRunStatus
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
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
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

        if adapter_key not in KNOWN_ADAPTERS:
            raise UnknownAdapterKeyError(adapter_key)

        if self._live_name_exists(name):
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

        self.db.add(source)
        self.db.flush()
        self.db.refresh(source)

        return self._to_source_view(source)

    def list_sources(self) -> list[SourceListItem]:
        sources = (
            self.db.query(IngestionSource)
            .filter(IngestionSource.deleted_at.is_(None))
            .order_by(IngestionSource.created_at.desc())
            .all()
        )

        stats_by_souce_id = self._stats_by_source_id([source.id for source in sources])

        return [
            SourceListItem(
                **self._to_source_view(source).model_dump(), stats=stats_by_souce_id[source.id]
            )
            for source in sources
        ]

    def get_source(self, source_id: int, *, recent_runs_limit: int = 20) -> SourceDetail:
        source = self._live_source_or_raise(source_id)

        runs = (
            self.db.query(SourceRun)
            .filter(SourceRun.source_id == source.id)
            .order_by(SourceRun.created_at.desc())
            .limit(recent_runs_limit)
            .all()
        )

        return SourceDetail(
            **self._to_source_view(source).model_dump(),
            stats=self._stats_by_source_id([source.id])[source.id],
            recent_runs=self._to_run_views(runs),
        )

    def patch(
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
        source = self._live_source_or_raise(source_id)

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

        self.db.flush()
        self.db.refresh(source)

        return self._to_source_view(source)

    def soft_delete(self, source_id: int) -> None:
        source = self._live_source_or_raise(source_id)

        source.deleted_at = datetime.datetime.now(datetime.UTC)
        self.db.flush()

    def _live_name_exists(self, name: str) -> bool:
        return (
            self.db.query(IngestionSource)
            .filter(IngestionSource.name == name, IngestionSource.deleted_at.is_(None))
            .first()
            is not None
        )

    def _live_source_or_raise(self, source_id: int) -> IngestionSource:
        source = (
            self.db.query(IngestionSource).filter(
                IngestionSource.id == source_id, IngestionSource.deleted_at.is_(None)
            )
        ).one_or_none()

        if source is None:
            raise SourceNotFoundError(source_id)

        return source

    def _stats_by_source_id(self, source_ids: list[int]) -> dict[int, SourceStats]:
        stats = {
            source_id: {"run_count": 0, "items_discovered": 0, "duplicate_count": 0}
            for source_id in source_ids
        }

        if not source_ids:
            return {}

        run_rows = self.db.execute(
            select(SourceRun.source_id, func.count(SourceRun.id))
            .where(SourceRun.source_id.in_(source_ids))
            .group_by(SourceRun.source_id)
        ).all()

        item_rows = self.db.execute(
            select(SourceItem.source_id, func.count(SourceItem.id))
            .where(SourceItem.source_id.in_(source_ids))
            .group_by(SourceItem.source_id)
        ).all()

        duplicate_rows = self.db.execute(
            select(IngestURL.source_id, func.count(IngestURL.id))
            .where(
                IngestURL.source_id.in_(source_ids),
                IngestURL.duplicate_reason.is_not(None),
            )
            .group_by(IngestURL.source_id)
        ).all()

        for source_id, count in run_rows:
            stats[source_id]["run_count"] = count

        for source_id, count in item_rows:
            stats[source_id]["items_discovered"] = count

        for source_id, count in duplicate_rows:
            stats[source_id]["duplicate_count"] = count

        return {source_id: SourceStats(**values) for source_id, values in stats.items()}

    def _to_run_views(self, runs: list[SourceRun]) -> list[SourceRunView]:
        run_ids = [run.id for run in runs]

        if not run_ids:
            return []

        discovered_by_run_id = self._discovered_count_by_run_id(run_ids)
        outcomes_by_run_id = self._url_outcomes_by_run_id(run_ids)

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

    def _discovered_count_by_run_id(self, run_ids: list[int]) -> dict[int, int]:
        rows = (
            self.db.query(SourceItem.last_source_run_id, func.count(SourceItem.id))
            .filter(SourceItem.last_source_run_id.in_(run_ids))
            .group_by(SourceItem.last_source_run_id)
            .all()
        )

        return {run_id: count for run_id, count in rows if run_id is not None}

    def _url_outcomes_by_run_id(self, run_ids: list[int]) -> dict[int, list[UrlOutcome]]:
        rows = (
            self.db.query(
                IngestURL.source_run_id,
                IngestURL.status,
                IngestURL.duplicate_reason,
            )
            .filter(IngestURL.source_run_id.in_(run_ids))
            .all()
        )

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
