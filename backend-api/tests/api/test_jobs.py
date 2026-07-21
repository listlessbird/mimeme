"""Tests for the /jobs endpoints, driven through the job feature interface.

The full application cannot be imported mid-rewrite (image/search routers and the
workflow package still reference not-yet-migrated code), so these tests mount only
the jobs router with the job-owned dependencies overridden.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.orm import Session

from mimeme.api.auth import ApiKeyRole, require_admin
from mimeme.api.deps import get_db, get_settings, get_temporal_client
from mimeme.api.routers.jobs import router
from mimeme.db.schema import Job, JobStatus, JobType, SearchIndexState
from tests.factories import create_index_build, create_job, create_search_index_state
from tests.job.conftest import SavepointDb


@pytest.fixture()
async def client(
    async_db_connection: AsyncConnection, mock_temporal: AsyncMock
) -> AsyncIterator[AsyncClient]:
    db = SavepointDb(async_db_connection)
    settings = SimpleNamespace(
        inference=SimpleNamespace(embed_model="siglip2-base"),
        index=SimpleNamespace(type="flat"),
        temporal=SimpleNamespace(task_queue="mimeme-v2"),
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: ApiKeyRole.ADMIN
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_temporal_client] = lambda: mock_temporal

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestGetJob:
    async def test_get_existing_job(self, client: AsyncClient, run_sync_seed) -> None:
        job_id = await run_sync_seed(lambda s: create_job(session=s).id)
        resp = await client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == JobStatus.PENDING.value

    async def test_ingest_json_result(self, client: AsyncClient, run_sync_seed) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, status=JobStatus.COMPLETED)
            job.result = json.dumps({"processed": 5, "failed": 0, "duplicates": 2})
            return job.id

        job_id = await run_sync_seed(seed)
        data = (await client.get(f"/jobs/{job_id}")).json()
        assert data["result"]["processed"] == 5 and data["result"]["duplicates"] == 2

    async def test_rebuild_json_result_shape(self, client: AsyncClient, run_sync_seed) -> None:
        def seed(session: Session) -> str:
            job = create_job(
                session=session, type=JobType.REBUILD_INDEX, status=JobStatus.COMPLETED
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
        data = (await client.get(f"/jobs/{job_id}")).json()
        assert data["result"] == {
            "version": "v-1",
            "num_vectors": 10,
            "dimension": 768,
            "removed_versions": ["v-old"],
            "text_num_vectors": 9,
            "skipped": False,
            "skip_reason": None,
        }

    async def test_invalid_json_result_is_raw(self, client: AsyncClient, run_sync_seed) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, status=JobStatus.COMPLETED)
            job.result = "not-json"
            return job.id

        job_id = await run_sync_seed(seed)
        assert (await client.get(f"/jobs/{job_id}")).json()["result"]["raw"] == "not-json"

    async def test_missing_returns_404(self, client: AsyncClient) -> None:
        assert (await client.get("/jobs/nope")).status_code == 404


class TestListJobs:
    async def test_empty(self, client: AsyncClient) -> None:
        data = (await client.get("/jobs")).json()
        assert data["jobs"] == [] and data["total"] == 0

    async def test_filter_by_status(self, client: AsyncClient, run_sync_seed) -> None:
        await run_sync_seed(
            lambda s: (
                create_job(session=s, status=JobStatus.PENDING),
                create_job(session=s, status=JobStatus.COMPLETED),
            )
        )
        data = (await client.get(f"/jobs?status={JobStatus.PENDING.value}")).json()
        assert data["total"] == 1 and data["jobs"][0]["status"] == JobStatus.PENDING.value

    async def test_filter_by_type(self, client: AsyncClient, run_sync_seed) -> None:
        await run_sync_seed(
            lambda s: (
                create_job(session=s, type=JobType.INGEST),
                create_job(session=s, type=JobType.REBUILD_INDEX),
            )
        )
        data = (await client.get(f"/jobs?job_type={JobType.INGEST.value}")).json()
        assert data["total"] == 1


class TestCancelJob:
    async def test_cancel_pending(
        self, client: AsyncClient, async_db_session, run_sync_seed
    ) -> None:
        job_id = await run_sync_seed(lambda s: create_job(session=s, status=JobStatus.PENDING).id)
        assert (await client.delete(f"/jobs/{job_id}")).status_code == 204
        assert (await async_db_session.get(Job, job_id)).status is JobStatus.CANCELLED

    async def test_cancel_running_cancels_workflow(
        self, client: AsyncClient, run_sync_seed, mock_temporal: AsyncMock
    ) -> None:
        job_id = await run_sync_seed(
            lambda s: create_job(session=s, status=JobStatus.RUNNING, workflow_id="wf-123").id
        )
        assert (await client.delete(f"/jobs/{job_id}")).status_code == 204
        mock_temporal.get_workflow_handle.assert_called_with("wf-123")

    async def test_cancel_without_workflow_skips_handle(
        self, client: AsyncClient, run_sync_seed, mock_temporal: AsyncMock
    ) -> None:
        job_id = await run_sync_seed(lambda s: create_job(session=s, status=JobStatus.PENDING).id)
        assert (await client.delete(f"/jobs/{job_id}")).status_code == 204
        mock_temporal.get_workflow_handle.assert_not_called()

    async def test_cancel_rebuild_releases_claim(
        self, client: AsyncClient, async_db_session, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, type=JobType.REBUILD_INDEX, status=JobStatus.RUNNING)
            session.flush()
            create_search_index_state(
                session=session,
                desired_generation=5,
                active_generation=1,
                rebuild_job_id=job.id,
                rebuild_target_generation=5,
                rebuild_claimed_at=datetime.datetime.now(datetime.UTC),
            )
            return job.id

        job_id = await run_sync_seed(seed)
        assert (await client.delete(f"/jobs/{job_id}")).status_code == 204
        state = await async_db_session.get(SearchIndexState, 1)
        assert state.rebuild_job_id is None and state.rebuild_target_generation is None

    @pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.FAILED])
    async def test_cancel_terminal_returns_400(
        self, client: AsyncClient, run_sync_seed, status: JobStatus
    ) -> None:
        job_id = await run_sync_seed(lambda s: create_job(session=s, status=status).id)
        assert (await client.delete(f"/jobs/{job_id}")).status_code == 400

    async def test_cancel_missing_returns_404(self, client: AsyncClient) -> None:
        assert (await client.delete("/jobs/nope")).status_code == 404


class TestRebuildIndex:
    async def test_trigger_creates_job(
        self, client: AsyncClient, async_db_session, mock_temporal: AsyncMock
    ) -> None:
        resp = await client.post("/jobs/rebuild-index")
        assert resp.status_code == 202
        data = resp.json()
        assert data["type"] == JobType.REBUILD_INDEX.value
        assert data["status"] == JobStatus.PENDING.value
        assert (await async_db_session.get(Job, data["id"])) is not None
        mock_temporal.start_workflow.assert_called_once()

    async def test_trigger_with_force(self, client: AsyncClient) -> None:
        assert (await client.post("/jobs/rebuild-index", json={"force": True})).status_code == 202


class TestIndexFreshness:
    async def test_stale_with_active_version(self, client: AsyncClient, run_sync_seed) -> None:
        def seed(session: Session) -> None:
            create_search_index_state(session=session, desired_generation=3, active_generation=1)
            create_index_build(session=session, version="v-cur", is_active=True)

        await run_sync_seed(seed)
        data = (await client.get("/jobs/indexes/freshness")).json()
        assert data["is_stale"] is True and data["active_version"] == "v-cur"

    async def test_current(self, client: AsyncClient, run_sync_seed) -> None:
        await run_sync_seed(
            lambda s: create_search_index_state(
                session=s, desired_generation=4, active_generation=4
            )
        )
        data = (await client.get("/jobs/indexes/freshness")).json()
        assert data["is_stale"] is False and data["active_version"] is None


class TestIndexVersions:
    async def test_empty(self, client: AsyncClient) -> None:
        assert (await client.get("/jobs/indexes/versions")).json()["versions"] == []

    async def test_with_data(self, client: AsyncClient, run_sync_seed) -> None:
        def seed(session: Session) -> None:
            create_index_build(session=session, is_active=True, version="v1")
            create_index_build(session=session, is_active=False, version="v2")

        await run_sync_seed(seed)
        versions = (await client.get("/jobs/indexes/versions")).json()["versions"]
        assert len(versions) == 2
        assert len([v for v in versions if v["is_active"]]) == 1
