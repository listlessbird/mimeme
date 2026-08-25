from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Sequence
from typing import Any

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mimeme.db import Db
from mimeme.db.schema import (
    Image,
    IngestionSource,
    IngestURL,
    Job,
    JobType,
    ProcessingStatus,
    SourceItem,
    SourceMedia,
    SourceRun,
    SourceRunStatus,
    SourceRunTrigger,
)
from mimeme.ingest.model import ItemRef, RemoteUrl, restore
from mimeme.job import rule as job_rule
from mimeme.media import Urls
from mimeme.source.model import (
    DiscoveredItem,
    DiscoveredMedia,
    DuplicateSourceName,
    RunItemsPage,
    RunItemView,
    RunNotFound,
    SourceDetail,
    SourceItemIngestState,
    SourceItemNotFound,
    SourceItemsPage,
    SourceItemView,
    SourceListItem,
    SourceNotFound,
    SourceRunView,
    SourceStats,
    SourceView,
    UrlOutcome,
    derive_run_accounting,
)
from mimeme.source.schedule import ScheduleSpec, derive_schedule_spec


class _Unset:
    pass


UNSET = _Unset()


class SourceConfig:
    """A read-only config snapshot the discover operation plans a fetch from."""

    def __init__(
        self,
        *,
        adapter_key: str,
        adapter_config: dict[str, Any],
        max_items_per_run: int | None,
        dataset: str | None,
    ) -> None:
        self.adapter_key = adapter_key
        self.adapter_config = adapter_config
        self.max_items_per_run = max_items_per_run
        self.dataset = dataset


class Store:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def live_name_exists(self, name: str) -> bool:
        return (
            await self._session.scalars(
                select(IngestionSource).where(
                    IngestionSource.name == name, IngestionSource.deleted_at.is_(None)
                )
            )
        ).first() is not None

    async def live_source_or_raise(self, source_id: int) -> IngestionSource:
        source = (
            await self._session.scalars(
                select(IngestionSource).where(
                    IngestionSource.id == source_id, IngestionSource.deleted_at.is_(None)
                )
            )
        ).one_or_none()
        if source is None:
            raise SourceNotFound(source_id)
        return source

    async def live_source_config(self, source_id: int) -> SourceConfig:
        source = await self.live_source_or_raise(source_id)
        return SourceConfig(
            adapter_key=source.adapter_key,
            adapter_config=dict(source.adapter_config),
            max_items_per_run=source.max_items_per_run,
            dataset=source.dataset,
        )

    async def insert_source(self, **fields: Any) -> SourceView:
        source = IngestionSource(**fields)
        self._session.add(source)
        await self._session.flush()
        await self._session.refresh(source)
        return _source_view(source)

    async def patch_source(self, source_id: int, changes: dict[str, Any]) -> SourceView:
        source = await self.live_source_or_raise(source_id)
        for field, value in changes.items():
            setattr(source, field, value)
        await self._session.flush()
        await self._session.refresh(source)
        return _source_view(source)

    async def soft_delete_source(self, source_id: int) -> None:
        source = await self.live_source_or_raise(source_id)
        source.deleted_at = datetime.datetime.now(datetime.UTC)
        await self._session.flush()

    async def list_source_items(self) -> list[SourceListItem]:
        sources = (
            await self._session.scalars(
                select(IngestionSource)
                .where(IngestionSource.deleted_at.is_(None))
                .order_by(IngestionSource.created_at.desc())
            )
        ).all()
        stats = await self._stats_by_source_id([source.id for source in sources])
        return [
            SourceListItem(**_source_view(source).model_dump(), stats=stats[source.id])
            for source in sources
        ]

    async def source_detail(self, source_id: int, *, recent_runs_limit: int) -> SourceDetail:
        source = await self.live_source_or_raise(source_id)
        runs = (
            await self._session.scalars(
                select(SourceRun)
                .where(SourceRun.source_id == source.id)
                .order_by(SourceRun.created_at.desc())
                .limit(recent_runs_limit)
            )
        ).all()
        stats = (await self._stats_by_source_id([source.id]))[source.id]
        return SourceDetail(
            **_source_view(source).model_dump(),
            stats=stats,
            recent_runs=await self._run_views(runs),
        )

    async def schedule_specs(self) -> list[ScheduleSpec]:
        sources = (
            await self._session.scalars(
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

    async def create_run(
        self,
        *,
        source_id: int,
        trigger: SourceRunTrigger,
        discovery_key: str | None = None,
    ) -> int:
        run = SourceRun(
            source_id=source_id,
            discovery_key=discovery_key,
            trigger_mode=trigger,
            status=SourceRunStatus.RUNNING,
            started_at=datetime.datetime.now(datetime.UTC),
        )
        self._session.add(run)
        await self._session.flush()
        return run.id

    async def replay_discovery(
        self, *, source_id: int, discovery_key: str
    ) -> tuple[int, str | None, list[ItemRef], int] | None:
        run = await self._session.scalar(
            select(SourceRun).where(
                SourceRun.source_id == source_id,
                SourceRun.discovery_key == discovery_key,
            )
        )
        if run is None:
            return None
        urls = (
            (
                await self._session.execute(
                    select(IngestURL)
                    .where(IngestURL.source_run_id == run.id)
                    .order_by(IngestURL.id.asc())
                )
            )
            .scalars()
            .all()
        )
        refs = [
            ItemRef(
                item_id=url.id,
                source=restore(
                    kind=url.input_kind,
                    url=url.url,
                    artifact_key=url.artifact_key,
                ),
            )
            for url in urls
        ]
        return run.id, run.ingest_job_id, refs, run.discovered_count

    async def get_run(self, source_run_id: int) -> SourceRun | None:
        return await self._session.get(SourceRun, source_run_id)

    async def seen_external_ids(self, source_id: int) -> set[str]:
        return {
            external_item_id
            for (external_item_id,) in (
                await self._session.execute(
                    select(SourceItem.external_item_id).where(SourceItem.source_id == source_id)
                )
            ).all()
        }

    async def touch_seen(
        self, *, source_id: int, source_run_id: int, external_ids: list[str]
    ) -> None:
        if not external_ids:
            return
        await self._session.execute(
            update(SourceItem)
            .where(
                SourceItem.source_id == source_id,
                SourceItem.external_item_id.in_(external_ids),
            )
            .values(
                last_seen_at=datetime.datetime.now(datetime.UTC),
                last_source_run_id=source_run_id,
            )
        )

    async def reconcile_discovery(
        self, *, source_id: int, source_run_id: int, items: list[DiscoveredItem]
    ) -> list[tuple[DiscoveredMedia, int, int]]:
        now = datetime.datetime.now(datetime.UTC)
        queued: list[tuple[DiscoveredMedia, int, int]] = []
        candidate_urls = {media.media_url for item in items for media in item.media}
        urls_seen_by_other_sources = (
            set(
                (
                    await self._session.scalars(
                        select(IngestURL.url).where(
                            IngestURL.url.in_(candidate_urls),
                            or_(IngestURL.source_id != source_id, IngestURL.source_id.is_(None)),
                        )
                    )
                ).all()
            )
            if candidate_urls
            else set()
        )
        for item in items:
            row = (
                await self._session.scalars(
                    select(SourceItem).where(
                        SourceItem.source_id == source_id,
                        SourceItem.external_item_id == item.external_item_id,
                    )
                )
            ).one_or_none()
            facts = item.known_facts.model_dump(mode="json")
            facts_hash = hashlib.sha256(
                json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            facts_changed = row is not None and row.known_facts_sha256 != facts_hash
            if row is None:
                row = SourceItem(
                    source_id=source_id,
                    external_item_id=item.external_item_id,
                    first_seen_at=now,
                )
                self._session.add(row)
            row.last_source_run_id = source_run_id
            row.canonical_item_url = item.canonical_item_url
            row.title = item.title
            row.known_facts = facts
            row.known_facts_sha256 = facts_hash
            row.raw_metadata = item.raw_metadata
            row.last_seen_at = now
            await self._session.flush()

            existing = {
                media.external_media_id: media
                for media in (
                    await self._session.scalars(
                        select(SourceMedia).where(SourceMedia.source_item_id == row.id)
                    )
                ).all()
            }
            for discovered_media in item.media:
                media = existing.get(discovered_media.external_media_id)
                is_new = media is None
                if media is None:
                    media = SourceMedia(
                        source_item_id=row.id,
                        external_media_id=discovered_media.external_media_id,
                        media_url=discovered_media.media_url,
                        first_seen_at=now,
                    )
                    self._session.add(media)
                media.media_url = discovered_media.media_url
                media.canonical_media_url = discovered_media.canonical_media_url
                media.raw_metadata = discovered_media.raw_metadata
                media.last_seen_at = now
                await self._session.flush()
                if (
                    is_new or facts_changed
                ) and discovered_media.media_url not in urls_seen_by_other_sources:
                    queued.append((discovered_media, row.id, media.id))
        run = await self._session.get(SourceRun, source_run_id)
        if run is not None:
            run.discovered_count = len(items)
            await self._session.flush()
        return queued

    async def create_ingest_job(
        self,
        *,
        source_id: int,
        source_run_id: int,
        media: list[tuple[DiscoveredMedia, int, int]],
    ) -> tuple[str, list[ItemRef]]:
        job_id, _ = job_rule.mint_ingest()
        self._session.add(Job(id=job_id, type=JobType.INGEST))
        await self._session.flush()
        refs: list[ItemRef] = []
        for item, source_item_id, source_media_id in media:
            url = IngestURL(
                job_id=job_id,
                input_kind="remote_image_url",
                url=item.media_url,
                source_id=source_id,
                source_run_id=source_run_id,
                source_item_id=source_item_id,
                source_media_id=source_media_id,
            )
            self._session.add(url)
            await self._session.flush()
            refs.append(ItemRef(item_id=url.id, source=RemoteUrl(url=item.media_url)))
        return job_id, refs

    async def link_run_ingest_job(self, *, source_run_id: int, ingest_job_id: str) -> None:
        run = await self._session.get(SourceRun, source_run_id)
        if run is not None:
            run.ingest_job_id = ingest_job_id
            await self._session.flush()

    async def discovered_count(self, source_run_id: int) -> int:
        return (
            await self._session.scalar(
                select(SourceRun.discovered_count).where(SourceRun.id == source_run_id)
            )
            or 0
        )

    async def url_outcomes(self, source_run_id: int) -> list[UrlOutcome]:
        rows = (
            await self._session.execute(
                select(IngestURL.status, IngestURL.duplicate_reason).where(
                    IngestURL.source_run_id == source_run_id
                )
            )
        ).all()
        return [
            UrlOutcome(status=status, duplicate_reason=duplicate_reason)
            for status, duplicate_reason in rows
        ]

    async def set_run_status(self, *, source_run_id: int, status: SourceRunStatus) -> None:
        run = await self._session.get(SourceRun, source_run_id)
        if run is None:
            return
        run.status = status
        if run.completed_at is None:
            run.completed_at = datetime.datetime.now(datetime.UTC)
        await self._session.flush()

    async def mark_run_failed(self, *, source_run_id: int, error: str) -> None:
        run = await self._session.get(SourceRun, source_run_id)
        if run is None:
            return
        run.status = SourceRunStatus.FAILED
        run.error_message = error[:1000]
        if run.completed_at is None:
            run.completed_at = datetime.datetime.now(datetime.UTC)
        await self._session.flush()

    async def failed_urls_for_run(self, source_id: int, run_id: int) -> Sequence[IngestURL]:
        run = (
            await self._session.execute(
                select(SourceRun).where(SourceRun.id == run_id, SourceRun.source_id == source_id)
            )
        ).scalar_one_or_none()
        if run is None:
            raise RunNotFound(run_id)
        return (
            (
                await self._session.execute(
                    select(IngestURL).where(
                        IngestURL.source_run_id == run_id,
                        IngestURL.status == ProcessingStatus.FAILED,
                    )
                )
            )
            .scalars()
            .all()
        )

    async def failed_urls_for_source(self, source_id: int) -> Sequence[IngestURL]:
        return (
            (
                await self._session.execute(
                    select(IngestURL).where(
                        IngestURL.source_id == source_id,
                        IngestURL.status == ProcessingStatus.FAILED,
                    )
                )
            )
            .scalars()
            .all()
        )

    async def failed_urls_for_item(
        self, source_id: int, source_item_id: int
    ) -> Sequence[IngestURL]:
        item = (
            await self._session.execute(
                select(SourceItem.id).where(
                    SourceItem.id == source_item_id, SourceItem.source_id == source_id
                )
            )
        ).scalar_one_or_none()
        if item is None:
            raise SourceItemNotFound(source_item_id)
        return (
            (
                await self._session.execute(
                    select(IngestURL).where(
                        IngestURL.source_item_id == source_item_id,
                        IngestURL.status == ProcessingStatus.FAILED,
                    )
                )
            )
            .scalars()
            .all()
        )

    async def reset_onto_new_job(
        self, urls: Sequence[IngestURL]
    ) -> tuple[str, list[int], list[ItemRef]]:
        job_id, _ = job_rule.mint_ingest()
        self._session.add(Job(id=job_id, type=JobType.INGEST))
        await self._session.flush()

        run_ids: list[int] = []
        refs: list[ItemRef] = []
        for url in urls:
            url.job_id = job_id
            url.status = ProcessingStatus.PENDING
            url.error_message = None
            url.image_id = None
            url.duplicate_reason = None
            url.duplicate_of_image_id = None
            url.similar_image_id = None
            url.phash_distance = None
            if url.source_run_id is not None and url.source_run_id not in run_ids:
                run_ids.append(url.source_run_id)
            refs.append(
                ItemRef(
                    item_id=url.id,
                    source=restore(kind=url.input_kind, url=url.url, artifact_key=url.artifact_key),
                )
            )

        for run_id in run_ids:
            run = await self._session.get(SourceRun, run_id)
            if run is not None:
                run.status = SourceRunStatus.RUNNING
        await self._session.flush()
        return job_id, run_ids, refs

    async def list_items(
        self,
        source_id: int,
        media_urls: Urls,
        *,
        limit: int,
        offset: int,
        status: SourceItemIngestState | None,
    ) -> SourceItemsPage:
        await self.live_source_or_raise(source_id)

        latest_attempt_id = (
            select(func.max(IngestURL.id))
            .where(IngestURL.source_item_id == SourceItem.id)
            .correlate(SourceItem)
            .scalar_subquery()
        )
        predicates = [SourceItem.source_id == source_id]
        if status is not None:
            predicates.append(_ingest_state_predicate(status))

        total = await self._session.scalar(
            select(func.count(SourceItem.id))
            .outerjoin(IngestURL, IngestURL.id == latest_attempt_id)
            .where(*predicates)
        )

        rows = (
            await self._session.execute(
                select(SourceItem, IngestURL, Image.s3_key)
                .outerjoin(IngestURL, IngestURL.id == latest_attempt_id)
                .outerjoin(
                    Image,
                    Image.id == func.coalesce(IngestURL.duplicate_of_image_id, IngestURL.image_id),
                )
                .where(*predicates)
                .order_by(SourceItem.last_seen_at.desc(), SourceItem.id.desc())
                .limit(limit)
                .offset(offset)
            )
        ).all()

        items: list[SourceItemView] = []
        for item, attempt, image_s3_key in rows:
            items.append(
                SourceItemView(
                    id=item.id,
                    external_item_id=item.external_item_id,
                    title=item.title,
                    raw_metadata=item.raw_metadata,
                    thumbnail_url=_thumbnail(
                        image_s3_key=image_s3_key,
                        raw_metadata=item.raw_metadata,
                        media_urls=media_urls,
                    ),
                    first_seen_at=item.first_seen_at,
                    last_seen_at=item.last_seen_at,
                    ingest_state=_derive_ingest_state(attempt),
                    resolved_image_id=_resolved_image_id(attempt),
                    duplicate_reason=attempt.duplicate_reason if attempt else None,
                    duplicate_of_image_id=attempt.duplicate_of_image_id if attempt else None,
                    attempt_status=attempt.status if attempt else None,
                    attempt_error_message=attempt.error_message if attempt else None,
                    attempt_source_run_id=attempt.source_run_id if attempt else None,
                    media_url=(
                        attempt.url
                        if attempt and attempt.input_kind == "remote_image_url"
                        else None
                    ),
                )
            )

        return SourceItemsPage(
            items=items,
            total=total or 0,
            limit=limit,
            offset=offset,
            state_counts=await self._state_counts(source_id),
        )

    async def list_run_items(
        self, source_id: int, run_id: int, media_urls: Urls, *, limit: int, offset: int
    ) -> RunItemsPage:
        await self.live_source_or_raise(source_id)
        run = (
            await self._session.execute(
                select(SourceRun.id).where(SourceRun.id == run_id, SourceRun.source_id == source_id)
            )
        ).scalar_one_or_none()
        if run is None:
            raise RunNotFound(run_id)

        total = await self._session.scalar(
            select(func.count(IngestURL.id)).where(IngestURL.source_run_id == run_id)
        )
        rows = (
            (
                await self._session.execute(
                    select(IngestURL)
                    .where(IngestURL.source_run_id == run_id)
                    .order_by(IngestURL.id.asc())
                    .limit(limit)
                    .offset(offset)
                )
            )
            .scalars()
            .all()
        )

        s3_key_by_image_id = await self._s3_keys_for(
            {
                resolved
                for url in rows
                if (resolved := url.image_id or url.duplicate_of_image_id) is not None
            }
        )
        item_by_id = await self._source_items_for(
            {url.source_item_id for url in rows if url.source_item_id is not None}
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
                    input=restore(kind=url.input_kind, url=url.url, artifact_key=url.artifact_key),
                    source_item_id=url.source_item_id,
                    external_item_id=source_item.external_item_id if source_item else None,
                    title=source_item.title if source_item else None,
                    status=url.status,
                    error_message=url.error_message,
                    duplicate_reason=url.duplicate_reason,
                    image_id=resolved_image_id,
                    thumbnail_url=_thumbnail(
                        image_s3_key=(
                            s3_key_by_image_id.get(resolved_image_id)
                            if resolved_image_id is not None
                            else None
                        ),
                        raw_metadata=source_item.raw_metadata if source_item else None,
                        media_urls=media_urls,
                    ),
                )
            )
        return RunItemsPage(items=items, total=total or 0, limit=limit, offset=offset)

    async def _stats_by_source_id(self, source_ids: list[int]) -> dict[int, SourceStats]:
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
            await self._session.execute(
                select(SourceRun.source_id, func.count(SourceRun.id))
                .where(SourceRun.source_id.in_(source_ids))
                .group_by(SourceRun.source_id)
            )
        ).all()
        item_rows = (
            await self._session.execute(
                select(SourceItem.source_id, func.count(SourceItem.id))
                .where(SourceItem.source_id.in_(source_ids))
                .group_by(SourceItem.source_id)
            )
        ).all()
        url_rows = (
            await self._session.execute(
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

    async def _run_views(self, runs: Sequence[SourceRun]) -> list[SourceRunView]:
        run_ids = [run.id for run in runs]
        if not run_ids:
            return []

        outcome_rows = (
            await self._session.execute(
                select(IngestURL.source_run_id, IngestURL.status, IngestURL.duplicate_reason).where(
                    IngestURL.source_run_id.in_(run_ids)
                )
            )
        ).all()
        outcomes_by_run: dict[int, list[UrlOutcome]] = {run_id: [] for run_id in run_ids}
        for run_id, status, duplicate_reason in outcome_rows:
            if run_id is not None:
                outcomes_by_run[run_id].append(
                    UrlOutcome(status=status, duplicate_reason=duplicate_reason)
                )

        views: list[SourceRunView] = []
        for run in runs:
            accounting = derive_run_accounting(
                discovered_items=run.discovered_count,
                url_outcomes=outcomes_by_run.get(run.id, []),
            )
            views.append(
                SourceRunView(
                    id=run.id,
                    trigger_mode=run.trigger_mode,
                    status=run.status,
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

    async def _state_counts(self, source_id: int) -> dict[str, int]:
        latest_attempt_id = (
            select(func.max(IngestURL.id))
            .where(IngestURL.source_item_id == SourceItem.id)
            .correlate(SourceItem)
            .scalar_subquery()
        )
        state = _state_case()
        rows = (
            await self._session.execute(
                select(state, func.count(SourceItem.id))
                .outerjoin(IngestURL, IngestURL.id == latest_attempt_id)
                .where(SourceItem.source_id == source_id)
                .group_by(state)
            )
        ).all()
        counts = {member.value: 0 for member in SourceItemIngestState}
        for label, count in rows:
            counts[label] = count
        return counts

    async def _s3_keys_for(self, image_ids: set[int]) -> dict[int, str | None]:
        if not image_ids:
            return {}
        rows = (
            await self._session.execute(
                select(Image.id, Image.s3_key).where(Image.id.in_(image_ids))
            )
        ).all()
        return {image_id: s3_key for image_id, s3_key in rows}

    async def _source_items_for(self, item_ids: set[int]) -> dict[int, SourceItem]:
        if not item_ids:
            return {}
        rows = (
            (await self._session.execute(select(SourceItem).where(SourceItem.id.in_(item_ids))))
            .scalars()
            .all()
        )
        return {item.id: item for item in rows}


def _source_view(source: IngestionSource) -> SourceView:
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


def _preview_from_metadata(raw_metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(raw_metadata, dict):
        return None
    preview = raw_metadata.get("preview")
    if isinstance(preview, list):
        urls = [item for item in preview if isinstance(item, str) and item]
        return urls[-1] if urls else None
    return preview if isinstance(preview, str) and preview else None


def _thumbnail(
    *, image_s3_key: str | None, raw_metadata: dict[str, Any] | None, media_urls: Urls
) -> str | None:
    if image_s3_key:
        return media_urls.resolve(image_s3_key)
    return _preview_from_metadata(raw_metadata)


def _state_case() -> Any:
    return case(
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


def _ingest_state_predicate(status: SourceItemIngestState) -> Any:
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


def _derive_ingest_state(attempt: IngestURL | None) -> SourceItemIngestState:
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


def _resolved_image_id(attempt: IngestURL | None) -> int | None:
    if attempt is None:
        return None
    if attempt.duplicate_reason is not None:
        return attempt.duplicate_of_image_id
    return attempt.image_id


# ---------------------------------------------------------------------------
# Db-scoped operations used by the API. Each opens its own session.
# ---------------------------------------------------------------------------


async def create(
    db: Db,
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
    from mimeme.source.adapter import KNOWN_ADAPTER_KEYS, UnknownAdapterKey

    if adapter_key not in KNOWN_ADAPTER_KEYS:
        raise UnknownAdapterKey(adapter_key)

    async with db.write_session() as session:
        store = Store(session)
        if await store.live_name_exists(name):
            raise DuplicateSourceName(name)
        return await store.insert_source(
            name=name,
            adapter_key=adapter_key,
            adapter_config=adapter_config,
            dataset=dataset,
            schedule_cron=schedule_cron,
            schedule_timezone=schedule_timezone,
            max_items_per_run=max_items_per_run,
            enabled=enabled,
        )


async def list_sources(db: Db) -> list[SourceListItem]:
    async with db.read_session() as session:
        return await Store(session).list_source_items()


async def get_source(db: Db, source_id: int, *, recent_runs_limit: int = 20) -> SourceDetail:
    async with db.read_session() as session:
        return await Store(session).source_detail(source_id, recent_runs_limit=recent_runs_limit)


async def patch(db: Db, source_id: int, changes: dict[str, Any]) -> SourceView:
    async with db.write_session() as session:
        return await Store(session).patch_source(source_id, changes)


async def soft_delete(db: Db, source_id: int) -> None:
    async with db.write_session() as session:
        await Store(session).soft_delete_source(source_id)


async def list_schedule_specs(db: Db) -> list[ScheduleSpec]:
    async with db.read_session() as session:
        return await Store(session).schedule_specs()


async def list_items(
    db: Db,
    source_id: int,
    media_urls: Urls,
    *,
    limit: int,
    offset: int,
    status: SourceItemIngestState | None = None,
) -> SourceItemsPage:
    async with db.read_session() as session:
        return await Store(session).list_items(
            source_id, media_urls, limit=limit, offset=offset, status=status
        )


async def list_run_items(
    db: Db, source_id: int, run_id: int, media_urls: Urls, *, limit: int, offset: int
) -> RunItemsPage:
    async with db.read_session() as session:
        return await Store(session).list_run_items(
            source_id, run_id, media_urls, limit=limit, offset=offset
        )
