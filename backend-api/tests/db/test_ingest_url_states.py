"""Tests for IngestURL model relationships and state transitions."""

from __future__ import annotations

from shared.models.orm import IngestURL, Job, ProcessingStatus
from tests.factories import ImageFactory, IngestURLFactory, JobFactory


class TestIngestURLRelationships:
    def test_multiple_urls_per_job(self, db_session) -> None:
        job = JobFactory(session=db_session)
        url1 = IngestURLFactory(session=db_session, job=job)
        url2 = IngestURLFactory(session=db_session, job=job)
        url3 = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        db_session.refresh(job)
        assert len(job.ingest_urls) == 3

    def test_cascade_delete_job_removes_urls(self, db_session) -> None:
        job = JobFactory(session=db_session)
        url1 = IngestURLFactory(session=db_session, job=job)
        url2 = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        job_id = job.id
        db_session.delete(job)
        db_session.flush()

        remaining = db_session.query(IngestURL).filter_by(job_id=job_id).all()
        assert len(remaining) == 0

    def test_image_fk_set_null_on_delete(self, db_session) -> None:
        """When an Image is deleted, IngestURL.image_id should be set to NULL."""
        job = JobFactory(session=db_session)
        image = ImageFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        url.image_id = image.id
        url.status = ProcessingStatus.DONE
        db_session.flush()

        db_session.delete(image)
        db_session.flush()

        db_session.refresh(url)
        assert url.image_id is None


class TestIngestURLStateTransitions:
    def test_default_status_is_pending(self, db_session) -> None:
        job = JobFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        db_session.flush()
        assert url.status == ProcessingStatus.PENDING

    def test_transition_to_done(self, db_session) -> None:
        job = JobFactory(session=db_session)
        image = ImageFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        url.status = ProcessingStatus.DONE
        url.image_id = image.id
        db_session.flush()

        db_session.refresh(url)
        assert url.status == ProcessingStatus.DONE
        assert url.image_id == image.id

    def test_transition_to_failed_with_error(self, db_session) -> None:
        job = JobFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        url.status = ProcessingStatus.FAILED
        url.error_message = "Connection timeout"
        db_session.flush()

        db_session.refresh(url)
        assert url.status == ProcessingStatus.FAILED
        assert url.error_message == "Connection timeout"

    def test_created_at_is_set(self, db_session) -> None:
        job = JobFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        db_session.flush()
        db_session.refresh(url)
        assert url.created_at is not None
