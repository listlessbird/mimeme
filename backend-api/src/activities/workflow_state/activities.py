from __future__ import annotations

import time

import structlog
from sqlalchemy import select
from temporalio import activity

from domain.job_lifecycle import JobLifecycle
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
    CreateRebuildJobInput,
    CreateRebuildJobOutput,
    FailRebuildJobInput,
    IngestInitOutput,
    IngestUrlItem,
    MarkIngestUrlDoneInput,
    MarkIngestUrlFailedInput,
    RecordIngestStageInput,
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
        with session_scope() as session:
            proc = session.scalars(
                select(Processing).where(Processing.image_id == input.image_id)
            ).first()
            found = proc is not None
            if proc:
                proc.embed_status = ProcessingStatus.DONE
                proc.embed_model = input.model
                proc.embed_dim = input.dimension
                proc.embed_s3_key = input.image_embedding_key
        emit_activity_event(
            log=log,
            activity_name="save_embedding_info_activity",
            started_at=started,
            outcome="success",
            image_id=input.image_id,
            found=found,
            dimension=input.dimension,
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
def create_rebuild_job_activity(input: CreateRebuildJobInput) -> CreateRebuildJobOutput:
    started = time.monotonic()
    try:
        with session_scope() as session:
            lifecycle = JobLifecycle(session)
            rebuild = lifecycle.create_rebuild_job(
                force=input.force,
                model_name=settings.embed_model,
                index_type=settings.index_type,
            )
            lifecycle.record_workflow_id(rebuild.job.id, rebuild.workflow_id)
            output = CreateRebuildJobOutput(
                job_id=rebuild.job.id,
                workflow_id=rebuild.workflow_id,
                force=rebuild.force,
                model_name=rebuild.model_name,
                index_type=rebuild.index_type,
            )
        emit_activity_event(
            log=log,
            activity_name="create_rebuild_job_activity",
            started_at=started,
            outcome="success",
            job_id=output.job_id,
            workflow_id=output.workflow_id,
            force=output.force,
            model_name=output.model_name,
            index_type=output.index_type,
        )
        return output
    except Exception as exc:
        emit_activity_event(
            log=log,
            activity_name="create_rebuild_job_activity",
            started_at=started,
            outcome="error",
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
