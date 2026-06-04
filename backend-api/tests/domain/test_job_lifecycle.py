from __future__ import annotations

import json

import pytest
from sqlalchemy.orm import Session
from tests.factories import create_image, create_ingest_url, create_job

from domain.job_lifecycle import (
    JobLifecycle,
    JobLifecycleInvalidStateError,
    JobLifecycleNotFoundError,
)
from shared.models.orm import DuplicateReason, IngestURL, JobStatus, JobType, ProcessingStatus


def test_create_ingest_job_deduplicates_urls_and_preserves_order(db_session: Session) -> None:
    result = JobLifecycle(db_session).create_ingest_job(
        urls=[
            "https://example.com/1.jpg",
            "https://example.com/1.jpg",
            "https://example.com/2.jpg",
        ],
        dataset="memes",
        tags=["funny"],
        callback_url="https://example.com/callback",
    )

    rows = db_session.query(IngestURL).filter_by(job_id=result.job_id).order_by(IngestURL.id).all()
    assert result.queued == 2
    assert result.duplicates == 1
    assert result.workflow_id == f"ingest-workflow-{result.job_id}"
    assert [row.url for row in rows] == [
        "https://example.com/1.jpg",
        "https://example.com/2.jpg",
    ]


def test_create_ingest_job_records_workflow_id(db_session: Session) -> None:
    lifecycle = JobLifecycle(db_session)
    result = lifecycle.create_ingest_job(
        urls=["https://example.com/1.jpg"],
        dataset=None,
        tags=[],
        callback_url=None,
    )

    lifecycle.record_workflow_id(result.job_id, result.workflow_id)

    assert lifecycle.get_job(result.job_id).id == result.job_id
    assert lifecycle.request_cancellation(result.job_id).workflow_id == result.workflow_id


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


def test_get_job_parses_json_result(db_session: Session) -> None:
    job = create_job(session=db_session, status=JobStatus.COMPLETED)
    job.result = json.dumps({"processed": 5, "failed": 0})
    db_session.flush()

    result = JobLifecycle(db_session).get_job(job.id)

    assert result.result == {"processed": 5, "failed": 0}


def test_get_job_preserves_invalid_json_as_raw_result(db_session: Session) -> None:
    job = create_job(session=db_session, status=JobStatus.COMPLETED)
    job.result = "not-json"
    db_session.flush()

    result = JobLifecycle(db_session).get_job(job.id)

    assert result.result == {"raw": "not-json"}


def test_get_missing_job_raises(db_session: Session) -> None:
    with pytest.raises(JobLifecycleNotFoundError):
        JobLifecycle(db_session).get_job("missing")


def test_list_jobs_filters_by_status_and_type(db_session: Session) -> None:
    create_job(session=db_session, type=JobType.INGEST, status=JobStatus.PENDING)
    create_job(session=db_session, type=JobType.REBUILD_INDEX, status=JobStatus.COMPLETED)
    db_session.flush()

    result = JobLifecycle(db_session).list_jobs(
        status=JobStatus.PENDING,
        job_type=JobType.INGEST,
        limit=20,
    )

    assert result.total == 1
    assert result.jobs[0].type == JobType.INGEST
    assert result.jobs[0].status == JobStatus.PENDING


def test_cancel_pending_job_without_workflow_id(db_session: Session) -> None:
    job = create_job(session=db_session, status=JobStatus.PENDING)
    db_session.flush()

    lifecycle = JobLifecycle(db_session)
    cancellation = lifecycle.request_cancellation(job.id)
    lifecycle.mark_cancelled(job.id)

    db_session.refresh(job)
    assert cancellation.workflow_id is None
    assert job.status == JobStatus.CANCELLED


def test_cancel_running_job_with_workflow_id(db_session: Session) -> None:
    job = create_job(session=db_session, status=JobStatus.RUNNING, workflow_id="wf-123")
    db_session.flush()

    cancellation = JobLifecycle(db_session).request_cancellation(job.id)

    assert cancellation.workflow_id == "wf-123"


@pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.FAILED])
def test_completed_and_failed_jobs_are_not_cancellable(
    db_session: Session,
    status: JobStatus,
) -> None:
    job = create_job(session=db_session, status=status)
    db_session.flush()

    with pytest.raises(JobLifecycleInvalidStateError):
        JobLifecycle(db_session).request_cancellation(job.id)


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
