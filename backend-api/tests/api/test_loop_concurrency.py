from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import pytest
from httpx import AsyncClient

from mimeme.db.schema import JobStatus, JobType
from mimeme.domain.job_rules import JobView
from mimeme.domain.job_store import ApiJobStore


async def test_concurrent_requests_overlap_on_the_loop(
    async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def slow_get_job(self: ApiJobStore, job_id: str) -> JobView:
        await asyncio.sleep(0.3)
        return JobView(
            id=job_id,
            type=JobType.INGEST,
            status=JobStatus.PENDING,
            progress=0.0,
            message=None,
            created_at=datetime.now(UTC),
            started_at=None,
            completed_at=None,
            result=None,
        )

    monkeypatch.setattr(ApiJobStore, "get_job", slow_get_job)

    start = time.perf_counter()
    first, second = await asyncio.gather(
        async_client.get("/jobs/ingest-aaaaaaaaaaaa"),
        async_client.get("/jobs/ingest-bbbbbbbbbbbb"),
    )
    elapsed = time.perf_counter() - start

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == "ingest-aaaaaaaaaaaa"
    assert second.json()["id"] == "ingest-bbbbbbbbbbbb"
    assert elapsed < 0.55
