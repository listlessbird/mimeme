"""Tests for the /jobs endpoints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from shared.models.orm import JobStatus, JobType
from tests.factories import create_index_build, create_job


class TestGetJob:
    def test_get_existing_job(self, client: TestClient, db_session: Session) -> None:
        job = create_job(session=db_session)
        db_session.flush()

        resp = client.get(f"/jobs/{job.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == job.id
        assert data["status"] == JobStatus.PENDING.value

    def test_get_job_with_json_result(self, client: TestClient, db_session: Session) -> None:
        job = create_job(session=db_session, status=JobStatus.COMPLETED)
        job.result = json.dumps({"processed": 5, "failed": 0})
        db_session.flush()

        resp = client.get(f"/jobs/{job.id}")
        data = resp.json()
        assert data["result"]["processed"] == 5

    def test_get_job_with_invalid_json_result(
        self, client: TestClient, db_session: Session
    ) -> None:
        job = create_job(session=db_session, status=JobStatus.COMPLETED)
        job.result = "not-valid-json"
        db_session.flush()

        resp = client.get(f"/jobs/{job.id}")
        data = resp.json()
        assert data["result"]["raw"] == "not-valid-json"

    def test_get_nonexistent_job_returns_404(self, client: TestClient) -> None:
        resp = client.get("/jobs/nonexistent")
        assert resp.status_code == 404


class TestListJobs:
    def test_list_jobs_empty(self, client: TestClient) -> None:
        resp = client.get("/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_list_jobs_with_data(self, client: TestClient, db_session: Session) -> None:
        create_job(session=db_session, type=JobType.INGEST)
        create_job(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()

        resp = client.get("/jobs")
        data = resp.json()
        assert data["total"] == 2

    def test_list_jobs_filter_by_status(self, client: TestClient, db_session: Session) -> None:
        create_job(session=db_session, status=JobStatus.PENDING)
        create_job(session=db_session, status=JobStatus.COMPLETED)
        db_session.flush()

        resp = client.get(f"/jobs?status={JobStatus.PENDING.value}")
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["status"] == JobStatus.PENDING.value

    def test_list_jobs_filter_by_type(self, client: TestClient, db_session: Session) -> None:
        create_job(session=db_session, type=JobType.INGEST)
        create_job(session=db_session, type=JobType.REBUILD_INDEX)
        db_session.flush()

        resp = client.get(f"/jobs?job_type={JobType.INGEST.value}")
        data = resp.json()
        assert data["total"] == 1


class TestCancelJob:
    def test_cancel_pending_job(
        self, client: TestClient, db_session: Session, mock_temporal: MagicMock
    ) -> None:
        job = create_job(session=db_session, status=JobStatus.PENDING)
        db_session.flush()

        resp = client.delete(f"/jobs/{job.id}")
        assert resp.status_code == 204

        db_session.refresh(job)
        assert job.status == JobStatus.CANCELLED

    def test_cancel_running_job_with_workflow(
        self, client: TestClient, db_session: Session, mock_temporal: MagicMock
    ) -> None:
        job = create_job(session=db_session, status=JobStatus.RUNNING, workflow_id="wf-123")
        db_session.flush()
        # Ensure the mock temporal's workflow_id attribute is set
        job.workflow_id = "wf-123"
        db_session.flush()

        resp = client.delete(f"/jobs/{job.id}")
        assert resp.status_code == 204
        mock_temporal.get_workflow_handle.assert_called_with("wf-123")

    def test_cancel_completed_job_returns_400(
        self, client: TestClient, db_session: Session
    ) -> None:
        job = create_job(session=db_session, status=JobStatus.COMPLETED)
        db_session.flush()

        resp = client.delete(f"/jobs/{job.id}")
        assert resp.status_code == 400

    def test_cancel_failed_job_returns_400(self, client: TestClient, db_session: Session) -> None:
        job = create_job(session=db_session, status=JobStatus.FAILED)
        db_session.flush()

        resp = client.delete(f"/jobs/{job.id}")
        assert resp.status_code == 400

    def test_cancel_nonexistent_job_returns_404(self, client: TestClient) -> None:
        resp = client.delete("/jobs/nonexistent")
        assert resp.status_code == 404

    def test_cancel_job_without_workflow_id(
        self, client: TestClient, db_session: Session, mock_temporal: MagicMock
    ) -> None:
        """Job with no workflow_id should still be cancellable."""
        job = create_job(session=db_session, status=JobStatus.PENDING)
        db_session.flush()

        resp = client.delete(f"/jobs/{job.id}")
        assert resp.status_code == 204
        mock_temporal.get_workflow_handle.assert_not_called()


class TestRebuildIndex:
    def test_trigger_rebuild_creates_job(
        self, client: TestClient, db_session: Session, mock_temporal: MagicMock
    ) -> None:
        resp = client.post("/jobs/rebuild-index")
        assert resp.status_code == 202
        data = resp.json()
        assert data["type"] == JobType.REBUILD_INDEX.value
        assert data["status"] == JobStatus.PENDING.value
        mock_temporal.start_workflow.assert_called_once()

    def test_trigger_rebuild_with_force(self, client: TestClient, mock_temporal: MagicMock) -> None:
        resp = client.post("/jobs/rebuild-index", json={"force": True})
        assert resp.status_code == 202


class TestIndexVersions:
    def test_list_index_versions_empty(self, client: TestClient) -> None:
        resp = client.get("/jobs/indexes/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["versions"] == []

    def test_list_index_versions_with_data(self, client: TestClient, db_session: Session) -> None:
        create_index_build(session=db_session, is_active=True, version="v1")
        create_index_build(session=db_session, is_active=False, version="v2")
        db_session.flush()

        resp = client.get("/jobs/indexes/versions")
        data = resp.json()
        assert len(data["versions"]) == 2
        active = [v for v in data["versions"] if v["is_active"]]
        assert len(active) == 1
