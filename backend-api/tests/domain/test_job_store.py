from __future__ import annotations

import pytest
from sqlalchemy.orm import Session
from tests.factories import create_job

from domain.job_lifecycle import JobLifecycleInvalidStateError, JobLifecycleNotFoundError
from domain.job_rules import IngestJobResultPayload, RawJobResultPayload, RebuildJobResultPayload
from domain.job_store import ApiJobStore
from shared.models import IngestURL, JobStatus, JobType

pytestmark = pytest.mark.usefixtures("_patch_domain_session_scope")


def test_create_ingest_job_deduplicates_urls_and_preserves_order(db_session: Session) -> None:
    result = ApiJobStore().create_ingest_job(
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


def test_create_rebuild_job_returns_pending_rebuild_view() -> None:
    result = ApiJobStore().create_rebuild_job(
        force=True,
        model_name="model-v2",
        index_type="flat",
    )

    assert result.job.type == JobType.REBUILD_INDEX
    assert result.job.status == JobStatus.PENDING
    assert result.workflow_id == f"rebuild-workflow-{result.job.id}"
    assert result.force is True
    assert result.model_name == "model-v2"


def test_create_ingest_job_records_workflow_id(db_session: Session) -> None:
    store = ApiJobStore()
    result = store.create_ingest_job(
        urls=["https://example.com/1.jpg"],
        dataset=None,
        tags=[],
        callback_url=None,
    )

    store.record_workflow_id(result.job_id, result.workflow_id)

    assert store.get_job(result.job_id).id == result.job_id
    assert store.request_cancellation(result.job_id).workflow_id == result.workflow_id


def test_get_job_returns_typed_result_payload(db_session: Session) -> None:
    job = create_job(session=db_session, status=JobStatus.COMPLETED)
    job.result = IngestJobResultPayload(processed=5, failed=0, duplicates=0).model_dump_json()
    db_session.flush()

    result = ApiJobStore().get_job(job.id)

    assert result.result == IngestJobResultPayload(processed=5, failed=0, duplicates=0)


def test_get_rebuild_job_returns_typed_result_payload(db_session: Session) -> None:
    job = create_job(session=db_session, type=JobType.REBUILD_INDEX, status=JobStatus.COMPLETED)
    job.result = RebuildJobResultPayload(
        version="v-1",
        num_vectors=10,
        dimension=768,
        removed_versions=["v-old"],
        text_num_vectors=9,
    ).model_dump_json()
    db_session.flush()

    result = ApiJobStore().get_job(job.id)

    assert result.result == RebuildJobResultPayload(
        version="v-1",
        num_vectors=10,
        dimension=768,
        removed_versions=["v-old"],
        text_num_vectors=9,
    )


def test_get_job_preserves_invalid_json_as_raw_result(db_session: Session) -> None:
    job = create_job(session=db_session, status=JobStatus.COMPLETED)
    job.result = "not-json"
    db_session.flush()

    result = ApiJobStore().get_job(job.id)

    assert result.result == RawJobResultPayload(raw="not-json")


def test_list_jobs_filters_by_status_and_type(db_session: Session) -> None:
    create_job(session=db_session, type=JobType.INGEST, status=JobStatus.PENDING)
    create_job(session=db_session, type=JobType.REBUILD_INDEX, status=JobStatus.COMPLETED)
    db_session.flush()

    result = ApiJobStore().list_jobs(
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

    store = ApiJobStore()
    cancellation = store.request_cancellation(job.id)
    store.mark_cancelled(job.id)

    db_session.refresh(job)
    assert cancellation.workflow_id is None
    assert job.status == JobStatus.CANCELLED


def test_cancel_running_job_with_workflow_id(db_session: Session) -> None:
    job = create_job(session=db_session, status=JobStatus.RUNNING, workflow_id="wf-123")
    db_session.flush()

    cancellation = ApiJobStore().request_cancellation(job.id)

    assert cancellation.workflow_id == "wf-123"


def test_completed_and_failed_jobs_are_not_cancellable(db_session: Session) -> None:
    for status in (JobStatus.COMPLETED, JobStatus.FAILED):
        job = create_job(session=db_session, status=status)
        db_session.flush()

        with pytest.raises(JobLifecycleInvalidStateError):
            ApiJobStore().request_cancellation(job.id)


def test_get_missing_job_raises() -> None:
    with pytest.raises(JobLifecycleNotFoundError):
        ApiJobStore().get_job("missing")
