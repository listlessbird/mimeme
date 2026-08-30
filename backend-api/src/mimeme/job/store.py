from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mimeme.db.schema import (
    Annotation,
    DuplicateReason,
    Image,
    IndexBuild,
    IngestStage,
    IngestURL,
    Job,
    JobStatus,
    JobType,
    Processing,
    ProcessingStatus,
    SearchIndexState,
    SourceItem,
)
from mimeme.ingest.model import restore
from mimeme.job import rule
from mimeme.job.model import (
    INGEST_URL_ERROR_LIMIT,
    JOB_ERROR_LIMIT,
    BuildView,
    ClaimOwnership,
    ClaimResult,
    ClaimTarget,
    EmbeddingSaved,
    Freshness,
    IngestInit,
    IngestResult,
    IngestUrlRef,
    ItemDone,
    NotFound,
    Page,
    RebuildResult,
    RowData,
    StateMissing,
    View,
)
from mimeme.search import document

_SINGLETON_ID = 1


def _freshness(row: SearchIndexState) -> Freshness:
    return Freshness(
        desired_generation=row.desired_generation,
        active_generation=row.active_generation,
        is_stale=row.desired_generation > row.active_generation,
        rebuild_job_id=row.rebuild_job_id,
        rebuild_target_generation=row.rebuild_target_generation,
        rebuild_claimed_at=row.rebuild_claimed_at,
        last_dirty_at=row.last_dirty_at,
        last_dirty_reason=row.last_dirty_reason,
        last_reconciled_at=row.last_reconciled_at,
    )


class Store:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_job(
        self, *, job_id: str, job_type: JobType, workflow_id: str | None = None
    ) -> None:
        self._session.add(Job(id=job_id, type=job_type, workflow_id=workflow_id))
        await self._session.flush()

    async def add_ingest_url(
        self,
        *,
        job_id: str,
        input_kind: str,
        url: str | None = None,
        artifact_key: str | None = None,
        source_id: int | None = None,
        source_run_id: int | None = None,
        source_item_id: int | None = None,
    ) -> None:
        self._session.add(
            IngestURL(
                job_id=job_id,
                input_kind=input_kind,
                url=url,
                artifact_key=artifact_key,
                source_id=source_id,
                source_run_id=source_run_id,
                source_item_id=source_item_id,
            )
        )
        await self._session.flush()

    async def view(self, job_id: str) -> View | None:
        job = await self._session.get(Job, job_id)
        if job is None:
            return None
        return rule.project(RowData.model_validate(job))

    async def page(self, *, status: JobStatus | None, job_type: JobType | None, limit: int) -> Page:
        stmt = select(Job)
        if status:
            stmt = stmt.where(Job.status == status)
        if job_type:
            stmt = stmt.where(Job.type == job_type)
        total = await self._session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = await self._session.scalars(stmt.order_by(Job.created_at.desc()).limit(limit))
        jobs = [rule.project(RowData.model_validate(job)) for job in rows.all()]
        return Page(jobs=jobs, total=total)

    async def index_builds(self, *, limit: int) -> list[BuildView]:
        rows = await self._session.scalars(
            select(IndexBuild).order_by(IndexBuild.created_at.desc()).limit(limit)
        )
        return [BuildView.model_validate(row) for row in rows.all()]

    async def active_version(self) -> str | None:
        return await self._session.scalar(
            select(IndexBuild.version).where(IndexBuild.is_active.is_(True))
        )

    async def active_build_stats(self) -> RebuildResult | None:
        row = (
            await self._session.scalars(select(IndexBuild).where(IndexBuild.is_active.is_(True)))
        ).first()
        if row is None:
            return None
        return RebuildResult(
            version=row.version,
            num_vectors=row.num_vectors or 0,
            dimension=row.dimension or 0,
            removed_versions=[],
            text_num_vectors=None,
            skipped=True,
            skip_reason="already_current",
        )

    async def set_workflow_id(self, job_id: str, workflow_id: str) -> None:
        job = await self._session.get(Job, job_id)
        if job is None:
            raise NotFound(f"Job {job_id} not found")
        job.workflow_id = workflow_id
        await self._session.flush()

    async def workflow_id_of(self, job_id: str) -> str | None:
        job = await self._session.get(Job, job_id)
        if job is None:
            raise NotFound(f"Job {job_id} not found")
        return job.workflow_id

    async def status_of(self, job_id: str) -> JobStatus:
        job = await self._session.get(Job, job_id)
        if job is None:
            raise NotFound(f"Job {job_id} not found")
        return job.status

    async def set_cancelled(self, job_id: str) -> None:
        job = await self._session.get(Job, job_id)
        if job is None:
            raise NotFound(f"Job {job_id} not found")
        job.status = JobStatus.CANCELLED
        await self._session.flush()

    async def mark_running(self, job_id: str) -> None:
        job = await self._session.get(Job, job_id)
        if job is None:
            raise NotFound(f"Job {job_id} not found")
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        await self._session.flush()

    async def ingest_urls(self, job_id: str) -> IngestInit:
        rows = (
            await self._session.scalars(select(IngestURL).where(IngestURL.job_id == job_id))
        ).all()
        return IngestInit(
            urls=[
                IngestUrlRef(
                    id=row.id,
                    input=restore(
                        kind=row.input_kind,
                        url=row.url,
                        artifact_key=row.artifact_key,
                    ),
                )
                for row in rows
            ]
        )

    async def set_progress(self, job_id: str, progress: float, message: str | None) -> bool:
        job = await self._session.get(Job, job_id)
        if job is None:
            return False
        job.progress = progress
        if message is not None:
            job.message = message
        await self._session.flush()
        return True

    async def complete_ingest(
        self, *, job_id: str, processed: int, failed: int, duplicates: int
    ) -> bool:
        job = await self._session.get(Job, job_id)
        if job is None:
            return False
        job.status = rule.derive_completion_status(failed=failed)
        job.progress = 100.0
        job.completed_at = datetime.now(UTC)
        job.result = IngestResult(
            processed=processed, failed=failed, duplicates=duplicates
        ).model_dump_json()
        await self._session.flush()
        return True

    async def fail_rebuild(self, job_id: str, error: str) -> bool:
        job = await self._session.get(Job, job_id)
        if job is None:
            return False
        job.status = JobStatus.FAILED
        job.message = rule.truncate(error, JOB_ERROR_LIMIT)
        job.completed_at = datetime.now(UTC)
        await self._session.flush()
        return True

    async def complete_rebuild(
        self,
        *,
        job_id: str,
        version: str,
        num_vectors: int,
        dimension: int,
        removed_versions: list[str],
        text_num_vectors: int | None,
        skipped: bool = False,
        skip_reason: str | None = None,
        message: str | None = None,
    ) -> None:
        job = await self._session.get(Job, job_id)
        if job is None:
            raise NotFound(f"Job {job_id} not found")
        job.status = JobStatus.COMPLETED
        job.progress = 100.0
        job.completed_at = datetime.now(UTC)
        if message is not None:
            job.message = message
        job.result = RebuildResult(
            version=version,
            num_vectors=num_vectors,
            dimension=dimension,
            removed_versions=removed_versions,
            text_num_vectors=text_num_vectors,
            skipped=skipped,
            skip_reason=skip_reason,
        ).model_dump_json()
        await self._session.flush()

    async def record_stage(self, ingest_url_id: int, stage: IngestStage) -> bool:
        url = await self._session.get(IngestURL, ingest_url_id)
        if url is None:
            return False
        url.stage = stage
        url.stage_updated_at = datetime.now(UTC)
        await self._session.flush()
        return True

    async def mark_item_failed(self, ingest_url_id: int, error: str) -> bool:
        url = await self._session.get(IngestURL, ingest_url_id)
        if url is None:
            return False
        url.status = ProcessingStatus.FAILED
        url.error_message = rule.truncate(error, INGEST_URL_ERROR_LIMIT)
        await self._session.flush()
        return True

    async def mark_item_done(
        self,
        ingest_url_id: int,
        image_id: int,
        duplicate_reason: DuplicateReason | None,
        duplicate_of_image_id: int | None,
        similar_image_id: int | None = None,
        phash_distance: int | None = None,
    ) -> ItemDone:
        url = await self._session.get(IngestURL, ingest_url_id)
        if url is None:
            return ItemDone(found=False, image_exists=None)
        prior_image_id = url.image_id
        image_exists = (
            await self._session.scalar(select(Image.id).where(Image.id == image_id))
        ) is not None
        if image_exists:
            before = await self._projected_document(image_id)
            url.status = ProcessingStatus.DONE
            url.image_id = image_id
            url.duplicate_reason = duplicate_reason
            url.duplicate_of_image_id = duplicate_of_image_id
            url.similar_image_id = similar_image_id
            url.phash_distance = phash_distance
            await self._session.flush()
            after = await self._projected_document(image_id)
            if (
                prior_image_id != image_id
                and before != after
                and await self._is_searchable(image_id)
            ):
                await self.mark_dirty(reason="source_alias_linked")
        else:
            url.status = ProcessingStatus.FAILED
            url.error_message = f"Image {image_id} not found while marking ingest URL done"
        await self._session.flush()
        return ItemDone(found=True, image_exists=image_exists)

    async def save_annotations(
        self,
        *,
        image_id: int,
        caption: str,
        caption_model: str,
        ocr_text: str,
        ocr_model: str,
        caption_context_sha256: str | None = None,
        caption_prompt_version: str | None = None,
    ) -> bool:
        # Duplicate ingestion batches can discover the same not-yet-annotated
        # image concurrently. Serialize the read-then-insert on its canonical
        # image row so both transactions cannot create the one annotation row.
        await self._session.scalar(select(Image.id).where(Image.id == image_id).with_for_update())
        ann = (
            await self._session.scalars(select(Annotation).where(Annotation.image_id == image_id))
        ).first()
        proc = (
            await self._session.scalars(select(Processing).where(Processing.image_id == image_id))
        ).first()
        before = document.project(
            image_id,
            caption=ann.caption_text if ann is not None else None,
            ocr_text=ann.ocr_text if ann is not None else None,
        )
        if ann is None:
            ann = Annotation(image_id=image_id)
            self._session.add(ann)
        ann.caption_text = caption
        ann.ocr_text = ocr_text
        ann.caption_context_sha256 = caption_context_sha256
        ann.caption_prompt_version = caption_prompt_version
        if proc is not None:
            proc.caption_status = ProcessingStatus.DONE
            proc.caption_model = caption_model
            proc.ocr_status = ProcessingStatus.DONE
            proc.ocr_model = ocr_model
        after = document.project(image_id, caption=caption, ocr_text=ocr_text)
        if (
            before != after
            and proc is not None
            and proc.embed_status is ProcessingStatus.DONE
            and bool(proc.embed_s3_key)
        ):
            await self.mark_dirty(reason="annotations_changed")
        await self._session.flush()
        return proc is not None

    async def _is_searchable(self, image_id: int) -> bool:
        return bool(
            await self._session.scalar(
                select(Processing.image_id).where(
                    Processing.image_id == image_id,
                    Processing.embed_status == ProcessingStatus.DONE,
                    Processing.embed_s3_key.is_not(None),
                    Processing.embed_s3_key != "",
                )
            )
        )

    async def _projected_document(self, image_id: int) -> document.SearchDocument:
        annotation = await self._session.get(Annotation, image_id)
        rows = (
            await self._session.execute(
                select(SourceItem.title, SourceItem.known_facts)
                .join(IngestURL, IngestURL.source_item_id == SourceItem.id)
                .where(IngestURL.image_id == image_id)
                .order_by(SourceItem.id)
            )
        ).all()
        return document.project(
            image_id,
            sources=(
                document.source_facts(
                    title,
                    facts if isinstance(facts, Mapping) else {},
                )
                for title, facts in rows
            ),
            caption=annotation.caption_text if annotation is not None else None,
            ocr_text=annotation.ocr_text if annotation is not None else None,
        )

    async def save_embedding(
        self,
        *,
        image_id: int,
        model: str,
        dimension: int,
        image_embedding_key: str,
        text_embedding_key: str | None,
        text_sha256: str | None = None,
        recipe_version: str | None = None,
    ) -> EmbeddingSaved:
        proc = (
            await self._session.scalars(select(Processing).where(Processing.image_id == image_id))
        ).first()
        if proc is None:
            return EmbeddingSaved(found=False, index_changed=False, desired_generation=None)
        before = (
            proc.embed_status,
            proc.embed_model,
            proc.embed_dim,
            proc.embed_s3_key,
            proc.embed_text_present,
            proc.embed_text_sha256,
            proc.embed_recipe_version,
        )
        superseded = (
            proc.embed_model != model
            or proc.embed_s3_key != image_embedding_key
            or proc.embed_text_sha256 != text_sha256
            or proc.embed_recipe_version != recipe_version
        )
        proc.embed_status = ProcessingStatus.DONE
        proc.embed_model = model
        proc.embed_dim = dimension
        proc.embed_s3_key = image_embedding_key
        proc.embed_text_present = bool(text_embedding_key)
        proc.embed_text_sha256 = text_sha256
        proc.embed_recipe_version = recipe_version
        if superseded:
            proc.embed_shard = None
            proc.embed_row = None
        after = (
            proc.embed_status,
            proc.embed_model,
            proc.embed_dim,
            proc.embed_s3_key,
            proc.embed_text_present,
            proc.embed_text_sha256,
            proc.embed_recipe_version,
        )
        if after == before:
            await self._session.flush()
            return EmbeddingSaved(found=True, index_changed=False, desired_generation=None)
        view = await self.mark_dirty(reason="embedding_saved")
        return EmbeddingSaved(
            found=True, index_changed=True, desired_generation=view.desired_generation
        )

    async def freshness(self) -> Freshness:
        return _freshness(await self._require())

    async def mark_dirty(self, *, reason: str) -> Freshness:
        stmt = (
            update(SearchIndexState)
            .where(SearchIndexState.id == _SINGLETON_ID)
            .values(
                desired_generation=SearchIndexState.desired_generation + 1,
                last_dirty_at=func.now(),
                last_dirty_reason=reason,
            )
            .returning(SearchIndexState)
            .execution_options(populate_existing=True)
        )
        row = (await self._session.execute(stmt)).scalars().one_or_none()
        if row is None:
            raise StateMissing("search_index_state singleton row is missing")
        return _freshness(row)

    async def claim(self, *, job_id: str, force: bool, now: datetime) -> ClaimResult:
        row = await self._require(lock=True)
        if row.rebuild_job_id is not None:
            return ClaimResult(acquired=False, reason="busy", view=_freshness(row))
        if not force and row.desired_generation <= row.active_generation:
            return ClaimResult(acquired=False, reason="clean", view=_freshness(row))
        row.rebuild_job_id = job_id
        row.rebuild_target_generation = row.desired_generation
        row.rebuild_claimed_at = now
        await self._session.flush()
        return ClaimResult(acquired=True, reason="acquired", view=_freshness(row))

    async def activate(
        self, *, job_id: str, target_generation: int, reconciled_at: datetime
    ) -> Freshness:
        row = await self._require(lock=True)
        if row.rebuild_job_id != job_id:
            raise ClaimOwnership(
                f"Job {job_id} does not own the rebuild claim (owner={row.rebuild_job_id})"
            )
        if row.rebuild_target_generation != target_generation:
            raise ClaimTarget(
                f"activation target {target_generation} does not match claim target "
                f"{row.rebuild_target_generation}"
            )
        row.active_generation = target_generation
        row.last_reconciled_at = reconciled_at
        await self._session.flush()
        return _freshness(row)

    async def release(self, *, job_id: str) -> Freshness:
        row = await self._require(lock=True)
        if row.rebuild_job_id is None:
            return _freshness(row)
        if row.rebuild_job_id != job_id:
            raise ClaimOwnership(
                f"Job {job_id} cannot release a claim owned by {row.rebuild_job_id}"
            )
        row.rebuild_job_id = None
        row.rebuild_target_generation = None
        row.rebuild_claimed_at = None
        await self._session.flush()
        return _freshness(row)

    async def lock_freshness(self) -> Freshness:
        return _freshness(await self._require(lock=True))

    async def _require(self, *, lock: bool = False) -> SearchIndexState:
        row = await self._session.get(SearchIndexState, _SINGLETON_ID, with_for_update=lock or None)
        if row is None:
            raise StateMissing("search_index_state singleton row is missing")
        return row
