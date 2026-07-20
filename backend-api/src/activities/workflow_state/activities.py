from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select
from temporalio import activity

from domain.index_freshness import (
    IndexFreshness,
    RebuildClaimOwnershipError,
    SearchIndexStateMissingError,
)
from domain.job_lifecycle import JobLifecycle
from domain.rebuild_claim import RebuildCoordinator
from shared.config import settings
from shared.db import session_scope
from shared.logging import emit_activity_event
from shared.models import (
    Annotation,
    Processing,
    ProcessingStatus,
)

from .models import (
    CompleteIngestJobInput,
    CompleteRebuildJobInput,
    FailRebuildJobInput,
    IngestInitOutput,
    IngestUrlItem,
    MarkIngestUrlDoneInput,
    MarkIngestUrlFailedInput,
    PrepareRebuildInput,
    PrepareRebuildOutput,
    ReconcileGenerationInput,
    RecordIngestStageInput,
    ReleaseRebuildClaimInput,
    SaveAnnotationsInput,
    SaveEmbeddingInfoInput,
    StartRebuildJobInput,
    UpdateJobProgressInput,
)

log = structlog.get_logger()


@activity.defn
def ingest_initialize_activity(job_id: str) -> IngestInitOutput:
    started = time.monotonic()
    try:
        with session_scope() as session:
            init = JobLifecycle(session).initialize_ingest(job_id)
            output = IngestInitOutput(
                urls=[IngestUrlItem(id=row.id, input=row.input) for row in init.urls]
            )
        emit_activity_event(
            log=log,
            activity_name="ingest_initialize_activity",
            started_at=started,
            outcome="success",
            job_id=job_id,
            url_count=len(output.urls),
        )
        return output
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="ingest_initialize_activity",
            started_at=started,
            outcome="error",
            job_id=job_id,
            error=str(exc),
        )
        raise


@activity.defn
def record_ingest_stage_activity(input: RecordIngestStageInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            found = JobLifecycle(session).record_stage(input.ingest_url_id, input.stage)
        emit_activity_event(
            log=log,
            activity_name="record_ingest_stage_activity",
            started_at=started,
            outcome="success",
            ingest_url_id=input.ingest_url_id,
            stage=input.stage.value,
            found=found,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="record_ingest_stage_activity",
            started_at=started,
            outcome="error",
            ingest_url_id=input.ingest_url_id,
            stage=input.stage.value,
            error=str(exc),
        )
        raise


@activity.defn
def mark_ingest_url_failed_activity(input: MarkIngestUrlFailedInput) -> None:
    started = time.monotonic()
    try:
        error_message = input.error[:1000] if input.error else input.error
        with session_scope() as session:
            found = JobLifecycle(session).mark_ingest_url_failed(
                input.ingest_url_id,
                input.error,
            )
        emit_activity_event(
            log=log,
            activity_name="mark_ingest_url_failed_activity",
            started_at=started,
            outcome="failed",
            ingest_url_id=input.ingest_url_id,
            found=found,
            failure_reason=error_message,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="mark_ingest_url_failed_activity",
            started_at=started,
            outcome="error",
            ingest_url_id=input.ingest_url_id,
            error=str(exc),
        )
        raise


@activity.defn
def mark_ingest_url_done_activity(input: MarkIngestUrlDoneInput) -> None:
    started = time.monotonic()
    try:
        image_exists: bool | None = None
        with session_scope() as session:
            result = JobLifecycle(session).mark_ingest_url_done(
                input.ingest_url_id,
                input.image_id,
                duplicate_reason=input.duplicate_reason,
                duplicate_of_image_id=input.duplicate_of_image_id,
            )
            found = result.found
            image_exists = result.image_exists
        emit_activity_event(
            log=log,
            activity_name="mark_ingest_url_done_activity",
            started_at=started,
            outcome="success" if (found and image_exists) else "failed",
            ingest_url_id=input.ingest_url_id,
            image_id=input.image_id,
            found=found,
            image_exists=image_exists,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="mark_ingest_url_done_activity",
            started_at=started,
            outcome="error",
            ingest_url_id=input.ingest_url_id,
            image_id=input.image_id,
            error=str(exc),
        )
        raise


@activity.defn
def save_annotations_activity(input: SaveAnnotationsInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            ann = session.scalars(
                select(Annotation).where(Annotation.image_id == input.image_id)
            ).first()
            proc = session.scalars(
                select(Processing).where(Processing.image_id == input.image_id)
            ).first()

            created = ann is None
            if not ann:
                ann = Annotation(image_id=input.image_id)
                session.add(ann)

            ann.caption_text = input.caption
            ann.ocr_text = input.ocr_text

            if proc:
                proc.caption_status = ProcessingStatus.DONE
                proc.caption_model = input.caption_model
                proc.ocr_status = ProcessingStatus.DONE
                proc.ocr_model = input.ocr_model
        emit_activity_event(
            log=log,
            activity_name="save_annotations_activity",
            started_at=started,
            outcome="success",
            image_id=input.image_id,
            annotation_created=created,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="save_annotations_activity",
            started_at=started,
            outcome="error",
            image_id=input.image_id,
            error=str(exc),
        )
        raise


@activity.defn
def save_embedding_info_activity(input: SaveEmbeddingInfoInput) -> None:
    started = time.monotonic()
    try:
        index_changed = False
        desired_generation: int | None = None
        with session_scope() as session:
            proc = session.scalars(
                select(Processing).where(Processing.image_id == input.image_id)
            ).first()
            found = proc is not None
            if proc:
                before = (proc.embed_status, proc.embed_model, proc.embed_dim, proc.embed_s3_key)
                proc.embed_status = ProcessingStatus.DONE
                proc.embed_model = input.model
                proc.embed_dim = input.dimension
                proc.embed_s3_key = input.image_embedding_key
                after = (proc.embed_status, proc.embed_model, proc.embed_dim, proc.embed_s3_key)
                if after != before:
                    view = IndexFreshness(session).mark_dirty(reason="embedding_saved")
                    index_changed = True
                    desired_generation = view.desired_generation
        emit_activity_event(
            log=log,
            activity_name="save_embedding_info_activity",
            started_at=started,
            outcome="success",
            image_id=input.image_id,
            found=found,
            dimension=input.dimension,
            index_changed=index_changed,
            desired_generation=desired_generation,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="save_embedding_info_activity",
            started_at=started,
            outcome="error",
            image_id=input.image_id,
            error=str(exc),
        )
        raise


@activity.defn
def update_job_progress_activity(input: UpdateJobProgressInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            found = JobLifecycle(session).update_progress(
                input.job_id,
                input.progress,
                input.message,
            )
        emit_activity_event(
            log=log,
            activity_name="update_job_progress_activity",
            started_at=started,
            outcome="success",
            job_id=input.job_id,
            found=found,
            progress=input.progress,
            has_message=input.message is not None,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="update_job_progress_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise


@activity.defn
def complete_ingest_job_activity(input: CompleteIngestJobInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            found = JobLifecycle(session).complete_ingest_job(
                job_id=input.job_id,
                processed=input.processed,
                failed=input.failed,
                duplicates=input.duplicates,
            )
        emit_activity_event(
            log=log,
            activity_name="complete_ingest_job_activity",
            started_at=started,
            outcome="failed" if input.failed > 0 else "success",
            job_id=input.job_id,
            found=found,
            processed=input.processed,
            failed=input.failed,
            duplicates=input.duplicates,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="complete_ingest_job_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise


@activity.defn
def prepare_rebuild_activity(input: PrepareRebuildInput) -> PrepareRebuildOutput:
    started = time.monotonic()
    try:
        with session_scope() as session:
            decision = RebuildCoordinator(session).prepare(
                job_id=input.job_id,
                workflow_id=input.workflow_id,
                force=input.force,
                trigger=input.trigger,
                now=datetime.now(UTC),
                claim_timeout=timedelta(
                    minutes=settings.search_index_rebuild_claim_timeout_minutes
                ),
            )
            output = PrepareRebuildOutput(
                decision=decision.decision,
                job_id=decision.job_id,
                target_generation=decision.target_generation,
            )
        emit_activity_event(
            log=log,
            activity_name="prepare_rebuild_activity",
            started_at=started,
            outcome="success",
            trigger=input.trigger.value,
            force=input.force,
            decision=output.decision,
            job_id=output.job_id,
            target_generation=output.target_generation,
        )
        return output
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="prepare_rebuild_activity",
            started_at=started,
            outcome="error",
            trigger=input.trigger.value,
            job_id=input.job_id,
            error=str(exc),
        )
        raise


@activity.defn
def reconcile_generation_activity(input: ReconcileGenerationInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            IndexFreshness(session).activate(
                job_id=input.job_id,
                target_generation=input.target_generation,
                reconciled_at=datetime.now(UTC),
            )
        emit_activity_event(
            log=log,
            activity_name="reconcile_generation_activity",
            started_at=started,
            outcome="success",
            job_id=input.job_id,
            target_generation=input.target_generation,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="reconcile_generation_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise


@activity.defn
def release_rebuild_claim_activity(input: ReleaseRebuildClaimInput) -> None:
    started = time.monotonic()
    released = True
    try:
        with session_scope() as session:
            try:
                IndexFreshness(session).release(job_id=input.job_id)
            except (RebuildClaimOwnershipError, SearchIndexStateMissingError):
                released = False
        emit_activity_event(
            log=log,
            activity_name="release_rebuild_claim_activity",
            started_at=started,
            outcome="success",
            job_id=input.job_id,
            released=released,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="release_rebuild_claim_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise


@activity.defn
def start_rebuild_job_activity(input: StartRebuildJobInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            JobLifecycle(session).start_job(input.job_id)
        emit_activity_event(
            log=log,
            activity_name="start_rebuild_job_activity",
            started_at=started,
            outcome="success",
            job_id=input.job_id,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="start_rebuild_job_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise


@activity.defn
def fail_rebuild_job_activity(input: FailRebuildJobInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            found = JobLifecycle(session).fail_rebuild_job(input.job_id, input.error)
        emit_activity_event(
            log=log,
            activity_name="fail_rebuild_job_activity",
            started_at=started,
            outcome="success",
            job_id=input.job_id,
            found=found,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="fail_rebuild_job_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise


@activity.defn
def complete_rebuild_job_activity(input: CompleteRebuildJobInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            JobLifecycle(session).complete_rebuild_job(
                job_id=input.job_id,
                version=input.version,
                num_vectors=input.num_vectors,
                dimension=input.dimension,
                removed_versions=input.removed_versions,
                text_num_vectors=input.text_num_vectors,
            )
        emit_activity_event(
            log=log,
            activity_name="complete_rebuild_job_activity",
            started_at=started,
            outcome="success",
            job_id=input.job_id,
            version=input.version,
            num_vectors=input.num_vectors,
            dimension=input.dimension,
            removed_versions=len(input.removed_versions),
            text_num_vectors=input.text_num_vectors,
        )
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="complete_rebuild_job_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise
