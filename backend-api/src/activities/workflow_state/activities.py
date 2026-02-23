from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import structlog
from temporalio import activity

from shared.db import session_scope
from shared.models import (
    Annotation,
    IngestURL,
    Job,
    JobStatus,
    ORMImage,
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
    SaveAnnotationsInput,
    SaveEmbeddingInfoInput,
    StartRebuildJobInput,
    UpdateJobProgressInput,
)

log = structlog.get_logger()


def _activity_context() -> dict[str, object]:
    try:
        info = activity.info()
    except RuntimeError:
        return {}
    return {
        "workflow_id": info.workflow_id,
        "run_id": info.workflow_run_id,
        "workflow_type": info.workflow_type,
        "activity_id": info.activity_id,
        "activity_type": info.activity_type,
        "attempt": info.attempt,
        "task_queue": info.task_queue,
        "is_local": info.is_local,
    }


def _emit_activity_event(
    *,
    activity_name: str,
    started_at: float,
    outcome: str,
    error: str | None = None,
    **fields: object,
) -> None:
    event: dict[str, object] = {
        "event_type": "activity_wide_event",
        "activity_name": activity_name,
        "outcome": outcome,
        "duration_ms": int((time.monotonic() - started_at) * 1000),
        **_activity_context(),
    }
    event.update(fields)
    if error:
        event["error"] = error
    log.info("activity_wide_event", **event)


@activity.defn
async def ingest_initialize_activity(job_id: str) -> IngestInitOutput:
    started = time.monotonic()
    try:
        with session_scope() as session:
            job = session.query(Job).filter_by(id=job_id).first()
            if not job:
                raise ValueError(f"Job {job_id} not found")

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)
            urls = session.query(IngestURL).filter_by(job_id=job_id).all()
            output = IngestInitOutput(urls=[IngestUrlItem(id=u.id, url=u.url) for u in urls])
        _emit_activity_event(
            activity_name="ingest_initialize_activity",
            started_at=started,
            outcome="success",
            job_id=job_id,
            url_count=len(output.urls),
        )
        return output
    except Exception as exc:
        _emit_activity_event(
            activity_name="ingest_initialize_activity",
            started_at=started,
            outcome="error",
            job_id=job_id,
            error=str(exc),
        )
        raise


@activity.defn
async def mark_ingest_url_failed_activity(input: MarkIngestUrlFailedInput) -> None:
    started = time.monotonic()
    try:
        error_message = input.error[:1000] if input.error else input.error
        with session_scope() as session:
            url = session.query(IngestURL).filter_by(id=input.ingest_url_id).first()
            found = url is not None
            if url:
                url.status = ProcessingStatus.FAILED
                url.error_message = error_message
        _emit_activity_event(
            activity_name="mark_ingest_url_failed_activity",
            started_at=started,
            outcome="failed",
            ingest_url_id=input.ingest_url_id,
            found=found,
            failure_reason=error_message,
        )
    except Exception as exc:
        _emit_activity_event(
            activity_name="mark_ingest_url_failed_activity",
            started_at=started,
            outcome="error",
            ingest_url_id=input.ingest_url_id,
            error=str(exc),
        )
        raise


@activity.defn
async def mark_ingest_url_done_activity(input: MarkIngestUrlDoneInput) -> None:
    started = time.monotonic()
    try:
        image_exists: bool | None = None
        with session_scope() as session:
            url = session.query(IngestURL).filter_by(id=input.ingest_url_id).first()
            found = url is not None
            if url:
                image_exists = (
                    session.query(ORMImage.id).filter(ORMImage.id == input.image_id).first() is not None
                )
                if image_exists:
                    url.status = ProcessingStatus.DONE
                    url.image_id = input.image_id
                else:
                    # Keep ingest row consistent even if upstream passed a stale/non-existent image id.
                    url.status = ProcessingStatus.FAILED
                    url.error_message = f"Image {input.image_id} not found while marking ingest URL done"
        _emit_activity_event(
            activity_name="mark_ingest_url_done_activity",
            started_at=started,
            outcome="success" if (found and image_exists) else "failed",
            ingest_url_id=input.ingest_url_id,
            image_id=input.image_id,
            found=found,
            image_exists=image_exists,
        )
    except Exception as exc:
        _emit_activity_event(
            activity_name="mark_ingest_url_done_activity",
            started_at=started,
            outcome="error",
            ingest_url_id=input.ingest_url_id,
            image_id=input.image_id,
            error=str(exc),
        )
        raise


@activity.defn
async def save_annotations_activity(input: SaveAnnotationsInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            ann = session.query(Annotation).filter_by(image_id=input.image_id).first()
            created = ann is None
            if not ann:
                ann = Annotation(image_id=input.image_id)
                session.add(ann)

            ann.caption_text = input.caption
            ann.ocr_text = input.ocr_text

            proc = session.query(Processing).filter_by(image_id=input.image_id).first()
            if proc:
                proc.caption_status = ProcessingStatus.DONE
                proc.caption_model = input.caption_model
                proc.ocr_status = ProcessingStatus.DONE
                proc.ocr_model = input.ocr_model
        _emit_activity_event(
            activity_name="save_annotations_activity",
            started_at=started,
            outcome="success",
            image_id=input.image_id,
            annotation_created=created,
        )
    except Exception as exc:
        _emit_activity_event(
            activity_name="save_annotations_activity",
            started_at=started,
            outcome="error",
            image_id=input.image_id,
            error=str(exc),
        )
        raise


@activity.defn
async def save_embedding_info_activity(input: SaveEmbeddingInfoInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            proc = session.query(Processing).filter_by(image_id=input.image_id).first()
            found = proc is not None
            if proc:
                proc.embed_status = ProcessingStatus.DONE
                proc.embed_model = input.model
                proc.embed_dim = input.dimension
                proc.embed_s3_key = input.image_embedding_key
        _emit_activity_event(
            activity_name="save_embedding_info_activity",
            started_at=started,
            outcome="success",
            image_id=input.image_id,
            found=found,
            dimension=input.dimension,
        )
    except Exception as exc:
        _emit_activity_event(
            activity_name="save_embedding_info_activity",
            started_at=started,
            outcome="error",
            image_id=input.image_id,
            error=str(exc),
        )
        raise


@activity.defn
async def update_job_progress_activity(input: UpdateJobProgressInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            job = session.query(Job).filter_by(id=input.job_id).first()
            found = job is not None
            if job:
                job.progress = input.progress
                if input.message is not None:
                    job.message = input.message
        _emit_activity_event(
            activity_name="update_job_progress_activity",
            started_at=started,
            outcome="success",
            job_id=input.job_id,
            found=found,
            progress=input.progress,
            has_message=input.message is not None,
        )
    except Exception as exc:
        _emit_activity_event(
            activity_name="update_job_progress_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise


@activity.defn
async def complete_ingest_job_activity(input: CompleteIngestJobInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            job = session.query(Job).filter_by(id=input.job_id).first()
            found = job is not None
            if job:
                job.status = JobStatus.COMPLETED if input.failed == 0 else JobStatus.FAILED
                job.progress = 100.0
                job.completed_at = datetime.now(UTC)
                job.result = json.dumps(
                    {
                        "processed": input.processed,
                        "failed": input.failed,
                        "duplicates": input.duplicates,
                    }
                )
        _emit_activity_event(
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
        _emit_activity_event(
            activity_name="complete_ingest_job_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise


@activity.defn
async def start_rebuild_job_activity(input: StartRebuildJobInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            job = session.query(Job).filter_by(id=input.job_id).first()
            if not job:
                raise ValueError(f"Job {input.job_id} not found")
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)
        _emit_activity_event(
            activity_name="start_rebuild_job_activity",
            started_at=started,
            outcome="success",
            job_id=input.job_id,
        )
    except Exception as exc:
        _emit_activity_event(
            activity_name="start_rebuild_job_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise


@activity.defn
async def fail_rebuild_job_activity(input: FailRebuildJobInput) -> None:
    started = time.monotonic()
    try:
        error_message = input.error[:2000] if input.error else input.error
        with session_scope() as session:
            job = session.query(Job).filter_by(id=input.job_id).first()
            found = job is not None
            if job:
                job.status = JobStatus.FAILED
                job.message = error_message
                job.completed_at = datetime.now(UTC)
        _emit_activity_event(
            activity_name="fail_rebuild_job_activity",
            started_at=started,
            outcome="success",
            job_id=input.job_id,
            found=found,
        )
    except Exception as exc:
        _emit_activity_event(
            activity_name="fail_rebuild_job_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise


@activity.defn
async def complete_rebuild_job_activity(input: CompleteRebuildJobInput) -> None:
    started = time.monotonic()
    try:
        with session_scope() as session:
            job = session.query(Job).filter_by(id=input.job_id).first()
            if not job:
                raise ValueError(f"Job {input.job_id} not found")
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = datetime.now(UTC)
            job.result = json.dumps(
                {
                    "version": input.version,
                    "num_vectors": input.num_vectors,
                    "dimension": input.dimension,
                    "removed_versions": input.removed_versions,
                    "text_num_vectors": input.text_num_vectors,
                }
            )
        _emit_activity_event(
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
        _emit_activity_event(
            activity_name="complete_rebuild_job_activity",
            started_at=started,
            outcome="error",
            job_id=input.job_id,
            error=str(exc),
        )
        raise
