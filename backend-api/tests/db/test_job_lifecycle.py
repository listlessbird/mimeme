"""Tests for Job ORM model state transitions and constraints."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError

from shared.models.orm import Job, JobStatus, JobType
from tests.factories import JobFactory


class TestJobDefaults:
    def test_default_status_is_pending(self, db_session) -> None:
        job = JobFactory(session=db_session)
        db_session.flush()
        assert job.status == JobStatus.PENDING

    def test_default_progress_is_zero(self, db_session) -> None:
        job = JobFactory(session=db_session)
        db_session.flush()
        assert job.progress == 0.0

    def test_created_at_is_set(self, db_session) -> None:
        job = JobFactory(session=db_session)
        db_session.flush()
        db_session.refresh(job)
        assert job.created_at is not None

    def test_started_at_and_completed_at_are_null(self, db_session) -> None:
        job = JobFactory(session=db_session)
        db_session.flush()
        assert job.started_at is None
        assert job.completed_at is None


class TestJobStateTransitions:
    def test_pending_to_running(self, db_session) -> None:
        job = JobFactory(session=db_session)
        db_session.flush()

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        db_session.flush()

        db_session.refresh(job)
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None

    def test_running_to_completed(self, db_session) -> None:
        job = JobFactory(session=db_session, status=JobStatus.RUNNING)
        db_session.flush()

        job.status = JobStatus.COMPLETED
        job.progress = 100.0
        job.completed_at = datetime.now(UTC)
        job.result = json.dumps({"processed": 5})
        db_session.flush()

        db_session.refresh(job)
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100.0
        result = json.loads(job.result)
        assert result["processed"] == 5

    def test_running_to_failed(self, db_session) -> None:
        job = JobFactory(session=db_session, status=JobStatus.RUNNING)
        db_session.flush()

        job.status = JobStatus.FAILED
        job.message = "Something went wrong"
        job.completed_at = datetime.now(UTC)
        db_session.flush()

        db_session.refresh(job)
        assert job.status == JobStatus.FAILED
        assert job.message == "Something went wrong"

    def test_pending_to_cancelled(self, db_session) -> None:
        job = JobFactory(session=db_session)
        db_session.flush()

        job.status = JobStatus.CANCELLED
        db_session.flush()

        db_session.refresh(job)
        assert job.status == JobStatus.CANCELLED


class TestJobConstraints:
    def test_duplicate_job_id_raises(self, db_session) -> None:
        job1 = JobFactory(session=db_session, id="dupe-id")
        db_session.flush()

        job2 = Job(id="dupe-id", type=JobType.INGEST)
        db_session.add(job2)
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_result_stores_valid_json(self, db_session) -> None:
        job = JobFactory(session=db_session)
        job.result = json.dumps({"key": "value", "nested": {"a": 1}})
        db_session.flush()

        db_session.refresh(job)
        parsed = json.loads(job.result)
        assert parsed["nested"]["a"] == 1

    def test_progress_boundary_values(self, db_session) -> None:
        job = JobFactory(session=db_session)
        db_session.flush()

        job.progress = 0.0
        db_session.flush()
        db_session.refresh(job)
        assert job.progress == 0.0

        job.progress = 100.0
        db_session.flush()
        db_session.refresh(job)
        assert job.progress == 100.0

    def test_both_job_types(self, db_session) -> None:
        j1 = JobFactory(session=db_session, type=JobType.INGEST)
        j2 = JobFactory(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()

        db_session.refresh(j1)
        db_session.refresh(j2)
        assert j1.type == JobType.INGEST
        assert j2.type == JobType.REBUILD_INDEX
