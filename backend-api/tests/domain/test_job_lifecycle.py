from __future__ import annotations

import json

from sqlalchemy.orm import Session
from tests.factories import create_image, create_ingest_url, create_job

from mimeme.db.schema import (
    DuplicateReason,
    IngestStage,
    JobStatus,
    JobType,
    ProcessingStatus,
)
from mimeme.domain.job_lifecycle import JobLifecycle


def test_create_rebuild_job_uses_rebuild_workflow_id(db_session: Session) -> None:
    result = JobLifecycle(db_session).create_rebuild_job(
        force=True,
        model_name="model-v2",
        index_type="flat",
    )

    assert result.job.type == JobType.REBUILD_INDEX
    assert result.workflow_id == f"rebuild-workflow-{result.job.id}"
    assert result.force is True
    assert result.model_name == "model-v2"


def test_initialize_ingest_starts_job_and_returns_urls(db_session: Session) -> None:
    job = create_job(session=db_session, type=JobType.INGEST)
    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()

    result = JobLifecycle(db_session).initialize_ingest(job.id)

    db_session.refresh(job)
    assert job.status == JobStatus.RUNNING
    assert job.started_at is not None
    assert result.urls[0].id == url.id


def test_update_progress_preserves_message_when_not_provided(db_session: Session) -> None:
    job = create_job(session=db_session, message="Old message")
    db_session.flush()

    found = JobLifecycle(db_session).update_progress(job.id, 50.0)

    db_session.refresh(job)
    assert found is True
    assert job.progress == 50.0
    assert job.message == "Old message"


def test_complete_ingest_job_with_failures_marks_failed(db_session: Session) -> None:
    job = create_job(session=db_session, status=JobStatus.RUNNING)
    db_session.flush()

    JobLifecycle(db_session).complete_ingest_job(
        job_id=job.id,
        processed=2,
        failed=1,
        duplicates=3,
    )

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED
    assert job.progress == 100.0
    assert json.loads(job.result or "{}") == {"processed": 2, "failed": 1, "duplicates": 3}


def test_fail_rebuild_job_truncates_error(db_session: Session) -> None:
    job = create_job(session=db_session, type=JobType.REBUILD_INDEX, status=JobStatus.RUNNING)
    db_session.flush()

    JobLifecycle(db_session).fail_rebuild_job(job.id, "x" * 5000)

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED
    assert job.message is not None
    assert len(job.message) == 2000


def test_complete_rebuild_job_stores_result(db_session: Session) -> None:
    job = create_job(session=db_session, type=JobType.REBUILD_INDEX, status=JobStatus.RUNNING)
    db_session.flush()

    JobLifecycle(db_session).complete_rebuild_job(
        job_id=job.id,
        version="v-1",
        num_vectors=10,
        dimension=768,
        removed_versions=["v-old"],
        text_num_vectors=9,
    )

    db_session.refresh(job)
    assert job.status == JobStatus.COMPLETED
    assert json.loads(job.result or "{}")["text_num_vectors"] == 9


def test_mark_ingest_url_done_with_missing_image_marks_failed(db_session: Session) -> None:
    job = create_job(session=db_session)
    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()

    result = JobLifecycle(db_session).mark_ingest_url_done(url.id, 999999)

    db_session.refresh(url)
    assert result.found is True
    assert result.image_exists is False
    assert url.status == ProcessingStatus.FAILED
    assert url.error_message is not None


def test_mark_ingest_url_done_with_existing_image(db_session: Session) -> None:
    job = create_job(session=db_session)
    image = create_image(session=db_session)
    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()

    result = JobLifecycle(db_session).mark_ingest_url_done(url.id, image.id)

    db_session.refresh(url)
    assert result.image_exists is True
    assert url.status == ProcessingStatus.DONE
    assert url.image_id == image.id


def test_mark_ingest_url_done_records_duplicate_provenance(db_session: Session) -> None:
    """A deduped URL points at the canonical image and records why."""
    job = create_job(session=db_session)
    canonical = create_image(session=db_session)
    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()

    JobLifecycle(db_session).mark_ingest_url_done(
        url.id,
        canonical.id,
        duplicate_reason=DuplicateReason.PHASH,
        duplicate_of_image_id=canonical.id,
    )

    db_session.refresh(url)
    assert url.status == ProcessingStatus.DONE
    assert url.image_id == canonical.id
    assert url.duplicate_reason == DuplicateReason.PHASH
    assert url.duplicate_of_image_id == canonical.id


def test_mark_ingest_url_done_leaves_provenance_null_for_new_image(db_session: Session) -> None:
    """A genuinely-new image records no duplicate provenance."""
    job = create_job(session=db_session)
    image = create_image(session=db_session)
    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()

    JobLifecycle(db_session).mark_ingest_url_done(url.id, image.id)

    db_session.refresh(url)
    assert url.duplicate_reason is None
    assert url.duplicate_of_image_id is None


def test_new_ingest_url_defaults_to_queued_stage(db_session: Session) -> None:
    job = create_job(session=db_session)
    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()

    db_session.refresh(url)
    assert url.stage == IngestStage.QUEUED
    assert url.stage_updated_at is None


def test_record_stage_sets_stage_and_timestamp(db_session: Session) -> None:
    job = create_job(session=db_session)
    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()

    found = JobLifecycle(db_session).record_stage(url.id, IngestStage.DOWNLOADING)

    db_session.refresh(url)
    assert found is True
    assert url.stage == IngestStage.DOWNLOADING
    assert url.stage_updated_at is not None


def test_record_stage_missing_url_returns_false(db_session: Session) -> None:
    assert JobLifecycle(db_session).record_stage(999999, IngestStage.DOWNLOADING) is False


def test_record_stage_advances_through_happy_path(db_session: Session) -> None:
    job = create_job(session=db_session)
    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()
    lifecycle = JobLifecycle(db_session)

    for stage in (
        IngestStage.DOWNLOADING,
        IngestStage.PROCESSING,
        IngestStage.ANNOTATING,
        IngestStage.EMBEDDING,
        IngestStage.COMPLETE,
    ):
        lifecycle.record_stage(url.id, stage)
        db_session.refresh(url)
        assert url.stage == stage


def test_record_stage_deduped_is_terminal(db_session: Session) -> None:
    job = create_job(session=db_session)
    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()
    lifecycle = JobLifecycle(db_session)

    lifecycle.record_stage(url.id, IngestStage.PROCESSING)
    lifecycle.record_stage(url.id, IngestStage.DEDUPED)

    db_session.refresh(url)
    assert url.stage == IngestStage.DEDUPED


def test_failure_freezes_stage_and_never_sets_failed(db_session: Session) -> None:
    """A failed attempt leaves `stage` frozen; failure lives in status/error."""
    job = create_job(session=db_session)
    url = create_ingest_url(session=db_session, job=job)
    db_session.flush()
    lifecycle = JobLifecycle(db_session)

    lifecycle.record_stage(url.id, IngestStage.ANNOTATING)
    lifecycle.mark_ingest_url_failed(url.id, "boom")

    db_session.refresh(url)
    assert url.stage == IngestStage.ANNOTATING
    assert url.status == ProcessingStatus.FAILED
    assert url.error_message == "boom"
