from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from mimeme.db import Db
from mimeme.db.schema import Job, JobStatus, JobType, RebuildTrigger
from mimeme.ingest.model import RemoteUrl, Source
from mimeme.job import rule
from mimeme.job.model import (
    BuildView,
    Cancellation,
    ClaimOwnership,
    EmbeddingSaved,
    Freshness,
    FreshnessStatus,
    IngestCreation,
    IngestInit,
    ItemDone,
    NotFound,
    Page,
    PrepareDecision,
    RebuildCreation,
    SourceIngestItem,
    StateMissing,
    View,
)
from mimeme.job.store import Store

_TERMINAL = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


async def create_ingest(
    db: Db,
    *,
    inputs: list[Source],
    dataset: str | None,
    tags: list[str],
    callback_url: str | None,
) -> IngestCreation:
    unique = list(dict.fromkeys(inputs))
    duplicates = len(inputs) - len(unique)
    job_id, workflow_id = rule.mint_ingest()
    async with db.write_session() as session:
        store = Store(session)
        await store.add_job(job_id=job_id, job_type=JobType.INGEST)
        for item in unique:
            if isinstance(item, RemoteUrl):
                await store.add_ingest_url(job_id=job_id, input_kind=item.kind, url=item.url)
            else:
                await store.add_ingest_url(
                    job_id=job_id, input_kind=item.kind, artifact_key=item.artifact_key
                )
    return IngestCreation(
        job_id=job_id,
        workflow_id=workflow_id,
        queued=len(unique),
        duplicates=duplicates,
        dataset=dataset,
        tags=tags,
        callback_url=callback_url,
    )


async def create_source_ingest(
    db: Db, *, dataset: str | None, items: list[SourceIngestItem]
) -> IngestCreation:
    job_id, workflow_id = rule.mint_ingest()
    async with db.write_session() as session:
        store = Store(session)
        await store.add_job(job_id=job_id, job_type=JobType.INGEST)
        for item in items:
            await store.add_ingest_url(
                job_id=job_id,
                input_kind="remote_image_url",
                url=item.url,
                source_id=item.source_id,
                source_run_id=item.source_run_id,
                source_item_id=item.source_item_id,
            )
    return IngestCreation(
        job_id=job_id,
        workflow_id=workflow_id,
        queued=len(items),
        duplicates=0,
        dataset=dataset,
        tags=[],
        callback_url=None,
    )


async def create_rebuild(
    db: Db, *, force: bool, model_name: str, index_type: Literal["flat", "hnsw"]
) -> RebuildCreation:
    job_id, workflow_id = rule.mint_rebuild()
    async with db.write_session() as session:
        store = Store(session)
        await store.add_job(job_id=job_id, job_type=JobType.REBUILD_INDEX)
        view = await store.view(job_id)
    assert view is not None
    return RebuildCreation(
        job=view,
        workflow_id=workflow_id,
        force=force,
        model_name=model_name,
        index_type=index_type,
    )


async def record_workflow_id(db: Db, job_id: str, workflow_id: str) -> None:
    async with db.write_session() as session:
        await Store(session).set_workflow_id(job_id, workflow_id)


async def find(db: Db, job_id: str) -> View | None:
    async with db.read_session() as session:
        return await Store(session).view(job_id)


async def find_exn(db: Db, job_id: str) -> View:
    view = await find(db, job_id)
    if view is None:
        raise NotFound(f"Job {job_id} not found")
    return view


async def list_jobs(
    db: Db, *, status: JobStatus | None, job_type: JobType | None, limit: int
) -> Page:
    async with db.read_session() as session:
        return await Store(session).page(status=status, job_type=job_type, limit=limit)


async def index_status(db: Db) -> FreshnessStatus:
    async with db.read_session() as session:
        store = Store(session)
        view = await store.freshness()
        active_version = await store.active_version()
    return FreshnessStatus(view=view, active_version=active_version)


async def list_index_builds(db: Db, *, limit: int) -> list[BuildView]:
    async with db.read_session() as session:
        return await Store(session).index_builds(limit=limit)


async def request_cancellation(db: Db, job_id: str) -> Cancellation:
    async with db.read_session() as session:
        store = Store(session)
        status = await store.status_of(job_id)
        rule.ensure_cancellable(status)
        return Cancellation(workflow_id=await store.workflow_id_of(job_id))


async def mark_cancelled(db: Db, job_id: str) -> None:
    async with db.write_session() as session:
        await Store(session).set_cancelled(job_id)


async def release_claim(db: Db, job_id: str) -> None:
    async with db.write_session() as session:
        try:
            await Store(session).release(job_id=job_id)
        except (ClaimOwnership, StateMissing):
            pass


async def initialize_ingest(db: Db, job_id: str) -> IngestInit:
    async with db.write_session() as session:
        store = Store(session)
        await store.mark_running(job_id)
        return await store.ingest_urls(job_id)


async def start(db: Db, job_id: str) -> None:
    async with db.write_session() as session:
        await Store(session).mark_running(job_id)


async def record_stage(db: Db, ingest_url_id: int, stage) -> bool:
    async with db.write_session() as session:
        return await Store(session).record_stage(ingest_url_id, stage)


async def mark_item_failed(db: Db, ingest_url_id: int, error: str) -> bool:
    async with db.write_session() as session:
        return await Store(session).mark_item_failed(ingest_url_id, error)


async def mark_item_done(
    db: Db,
    ingest_url_id: int,
    image_id: int,
    *,
    duplicate_reason=None,
    duplicate_of_image_id: int | None = None,
    similar_image_id: int | None = None,
    phash_distance: int | None = None,
) -> ItemDone:
    async with db.write_session() as session:
        return await Store(session).mark_item_done(
            ingest_url_id,
            image_id,
            duplicate_reason,
            duplicate_of_image_id,
            similar_image_id,
            phash_distance,
        )


async def save_annotations(
    db: Db,
    *,
    image_id: int,
    caption: str,
    caption_model: str,
    ocr_text: str,
    ocr_model: str,
    caption_context_sha256: str | None = None,
    caption_prompt_version: str | None = None,
) -> bool:
    async with db.write_session() as session:
        return await Store(session).save_annotations(
            image_id=image_id,
            caption=caption,
            caption_model=caption_model,
            ocr_text=ocr_text,
            ocr_model=ocr_model,
            caption_context_sha256=caption_context_sha256,
            caption_prompt_version=caption_prompt_version,
        )


async def save_embedding(
    db: Db,
    *,
    image_id: int,
    model: str,
    dimension: int,
    image_embedding_key: str,
    text_embedding_key: str | None,
    text_sha256: str | None = None,
    recipe_version: str | None = None,
) -> EmbeddingSaved:
    async with db.write_session() as session:
        return await Store(session).save_embedding(
            image_id=image_id,
            model=model,
            dimension=dimension,
            image_embedding_key=image_embedding_key,
            text_embedding_key=text_embedding_key,
            text_sha256=text_sha256,
            recipe_version=recipe_version,
        )


async def progress(db: Db, job_id: str, value: float, message: str | None = None) -> bool:
    async with db.write_session() as session:
        return await Store(session).set_progress(job_id, value, message)


async def complete_ingest(
    db: Db, *, job_id: str, processed: int, failed: int, duplicates: int
) -> bool:
    async with db.write_session() as session:
        return await Store(session).complete_ingest(
            job_id=job_id, processed=processed, failed=failed, duplicates=duplicates
        )


async def fail_rebuild(db: Db, job_id: str, error: str) -> bool:
    async with db.write_session() as session:
        return await Store(session).fail_rebuild(job_id, error)


async def complete_rebuild(
    db: Db,
    *,
    job_id: str,
    version: str,
    num_vectors: int,
    dimension: int,
    removed_versions: list[str],
    text_num_vectors: int | None,
) -> None:
    async with db.write_session() as session:
        await Store(session).complete_rebuild(
            job_id=job_id,
            version=version,
            num_vectors=num_vectors,
            dimension=dimension,
            removed_versions=removed_versions,
            text_num_vectors=text_num_vectors,
        )


async def activate_generation(
    db: Db, *, job_id: str, target_generation: int, reconciled_at: datetime
) -> Freshness:
    async with db.write_session() as session:
        return await Store(session).activate(
            job_id=job_id, target_generation=target_generation, reconciled_at=reconciled_at
        )


async def prepare_rebuild(
    db: Db,
    *,
    job_id: str | None,
    workflow_id: str,
    force: bool,
    trigger: RebuildTrigger,
    now: datetime,
    claim_timeout: timedelta,
) -> PrepareDecision:
    async with db.write_session() as session:
        store = Store(session)
        view = await store.lock_freshness()

        claim = view.active_claim
        if claim is not None:
            owner = await session.get(Job, claim.job_id)
            if owner is None or owner.status in _TERMINAL:
                await store.release(job_id=claim.job_id)
                view = await store.lock_freshness()
            elif claim.claimed_at + claim_timeout <= now:
                await store.fail_rebuild(
                    claim.job_id, f"Rebuild claim expired; reclaimed by {trigger.value}"
                )
                await store.release(job_id=claim.job_id)
                view = await store.lock_freshness()

        if view.active_claim is not None:
            return PrepareDecision(
                decision="busy",
                job_id=job_id if trigger is RebuildTrigger.MANUAL else None,
            )

        if not view.is_stale and not force:
            if trigger is RebuildTrigger.MANUAL and job_id is not None:
                active = await store.active_build_stats()
                await store.complete_rebuild(
                    job_id=job_id,
                    version=active.version if active else "",
                    num_vectors=active.num_vectors if active else 0,
                    dimension=active.dimension if active else 0,
                    removed_versions=[],
                    text_num_vectors=None,
                    skipped=True,
                    skip_reason="already_current",
                    message="Index already current",
                )
            return PrepareDecision(
                decision="clean",
                job_id=job_id if trigger is RebuildTrigger.MANUAL else None,
            )

        if trigger is RebuildTrigger.SCHEDULED:
            job_id, _ = rule.mint_rebuild()
            await store.add_job(
                job_id=job_id, job_type=JobType.REBUILD_INDEX, workflow_id=workflow_id
            )

        assert job_id is not None
        result = await store.claim(job_id=job_id, force=force, now=now)
        return PrepareDecision(
            decision="build",
            job_id=job_id,
            target_generation=result.view.rebuild_target_generation,
        )
