"""Tests for the /jobs endpoints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from shared.models.orm import Job, JobStatus, JobType
from tests.factories import create_index_build, create_job


class TestGetJob:
    async def test_get_existing_job(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        job_id = await run_sync_seed(lambda session: create_job(session=session).id)

        resp = await async_client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == job_id
        assert data["status"] == JobStatus.PENDING.value

    async def test_get_job_with_json_result(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        def seed(session) -> str:
            job = create_job(session=session, status=JobStatus.COMPLETED)
            job.result = json.dumps({"processed": 5, "failed": 0, "duplicates": 0})
            return job.id

        job_id = await run_sync_seed(seed)

        resp = await async_client.get(f"/jobs/{job_id}")
        data = resp.json()
        assert data["result"]["processed"] == 5
        assert data["result"]["duplicates"] == 0

    async def test_get_rebuild_job_with_json_result(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        def seed(session) -> str:
            job = create_job(
                session=session,
                type=JobType.REBUILD_INDEX,
                status=JobStatus.COMPLETED,
            )
            job.result = json.dumps(
                {
                    "version": "v-1",
                    "num_vectors": 10,
                    "dimension": 768,
                    "removed_versions": ["v-old"],
                    "text_num_vectors": 9,
                }
            )
            return job.id

        job_id = await run_sync_seed(seed)

        resp = await async_client.get(f"/jobs/{job_id}")
        data = resp.json()

        assert data["result"] == {
            "version": "v-1",
            "num_vectors": 10,
            "dimension": 768,
            "removed_versions": ["v-old"],
            "text_num_vectors": 9,
        }

    async def test_get_job_with_invalid_json_result(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        def seed(session) -> str:
            job = create_job(session=session, status=JobStatus.COMPLETED)
            job.result = "not-valid-json"
            return job.id

        job_id = await run_sync_seed(seed)

        resp = await async_client.get(f"/jobs/{job_id}")
        data = resp.json()
        assert data["result"]["raw"] == "not-valid-json"

    async def test_get_nonexistent_job_returns_404(
        self, async_client: AsyncClient, _patch_async_domain_session_scope: None
    ) -> None:
        resp = await async_client.get("/jobs/nonexistent")
        assert resp.status_code == 404


class TestListJobs:
    async def test_list_jobs_empty(
        self, async_client: AsyncClient, _patch_async_domain_session_scope: None
    ) -> None:
        resp = await async_client.get("/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    async def test_list_jobs_with_data(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        await run_sync_seed(
            lambda session: (
                create_job(session=session, type=JobType.INGEST),
                create_job(session=session, type=JobType.REBUILD_INDEX),
            )
        )

        resp = await async_client.get("/jobs")
        data = resp.json()
        assert data["total"] == 2

    async def test_list_jobs_filter_by_status(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        await run_sync_seed(
            lambda session: (
                create_job(session=session, status=JobStatus.PENDING),
                create_job(session=session, status=JobStatus.COMPLETED),
            )
        )

        resp = await async_client.get(f"/jobs?status={JobStatus.PENDING.value}")
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["status"] == JobStatus.PENDING.value

    async def test_list_jobs_filter_by_type(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        await run_sync_seed(
            lambda session: (
                create_job(session=session, type=JobType.INGEST),
                create_job(session=session, type=JobType.REBUILD_INDEX),
            )
        )

        resp = await async_client.get(f"/jobs?job_type={JobType.INGEST.value}")
        data = resp.json()
        assert data["total"] == 1


class TestCancelJob:
    async def test_cancel_pending_job(
        self,
        async_client: AsyncClient,
        async_db_session: AsyncSession,
        run_sync_seed,
        mock_temporal: MagicMock,
        _patch_async_domain_session_scope: None,
    ) -> None:
        job_id = await run_sync_seed(
            lambda session: create_job(session=session, status=JobStatus.PENDING).id
        )

        resp = await async_client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 204

        job = await async_db_session.get(Job, job_id)
        assert job is not None
        assert job.status == JobStatus.CANCELLED

    async def test_cancel_running_job_with_workflow(
        self,
        async_client: AsyncClient,
        run_sync_seed,
        mock_temporal: MagicMock,
        _patch_async_domain_session_scope: None,
    ) -> None:
        job_id = await run_sync_seed(
            lambda session: (
                create_job(session=session, status=JobStatus.RUNNING, workflow_id="wf-123").id
            )
        )

        resp = await async_client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 204
        mock_temporal.get_workflow_handle.assert_called_with("wf-123")

    async def test_cancel_completed_job_returns_400(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        job_id = await run_sync_seed(
            lambda session: create_job(session=session, status=JobStatus.COMPLETED).id
        )

        resp = await async_client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 400

    async def test_cancel_failed_job_returns_400(
        self, async_client: AsyncClient, run_sync_seed, _patch_async_domain_session_scope: None
    ) -> None:
        job_id = await run_sync_seed(
            lambda session: create_job(session=session, status=JobStatus.FAILED).id
        )

        resp = await async_client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 400

    async def test_cancel_nonexistent_job_returns_404(
        self, async_client: AsyncClient, _patch_async_domain_session_scope: None
    ) -> None:
        resp = await async_client.delete("/jobs/nonexistent")
        assert resp.status_code == 404

    async def test_cancel_job_without_workflow_id(
        self,
        async_client: AsyncClient,
        run_sync_seed,
        mock_temporal: MagicMock,
        _patch_async_domain_session_scope: None,
    ) -> None:
        job_id = await run_sync_seed(
            lambda session: create_job(session=session, status=JobStatus.PENDING).id
        )

        resp = await async_client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 204
        mock_temporal.get_workflow_handle.assert_not_called()


class TestRebuildIndex:
    async def test_trigger_rebuild_creates_job(
        self,
        async_client: AsyncClient,
        async_db_session: AsyncSession,
        mock_temporal: MagicMock,
        _patch_async_domain_session_scope: None,
    ) -> None:
        resp = await async_client.post("/jobs/rebuild-index")
        assert resp.status_code == 202
        data = resp.json()
        assert data["type"] == JobType.REBUILD_INDEX.value
        assert data["status"] == JobStatus.PENDING.value
        job_status = await async_db_session.scalar(select(Job.status).where(Job.id == data["id"]))
        assert job_status == JobStatus.PENDING
        mock_temporal.start_workflow.assert_called_once()

    async def test_trigger_rebuild_with_force(
        self,
        async_client: AsyncClient,
        mock_temporal: MagicMock,
        _patch_async_domain_session_scope: None,
    ) -> None:
        resp = await async_client.post("/jobs/rebuild-index", json={"force": True})
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
