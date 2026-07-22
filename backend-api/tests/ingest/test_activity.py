from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.orm import Session
from temporalio.testing import ActivityEnvironment

from mimeme.db.schema import Job, JobStatus, ProcessingStatus
from mimeme.ingest import rule
from mimeme.ingest.activity import IngestActivities
from mimeme.ingest.model import Finish, Input, RemoteUrl
from tests.factories import (
    create_ingest_url,
    create_job,
    create_search_index_state,
)
from tests.ingest.conftest import FakeEnv, FakeImages, FakeInference, facts_for, image_http
from tests.job.conftest import SavepointDb

REMOTE_URL = "https://example.com/cat.png"


def test_activity_names() -> None:
    acts = IngestActivities(FakeEnv(db=None))  # type: ignore[arg-type]
    assert IngestActivities.item.__temporal_activity_definition.name == rule.ITEM_ACTIVITY
    assert IngestActivities.finish.__temporal_activity_definition.name == rule.FINISH_ACTIVITY
    del acts


class TestItem:
    async def test_item_processes_and_updates_progress(
        self, db: SavepointDb, run_sync_seed, png_bytes: bytes
    ) -> None:
        def seed(s: Session) -> tuple[str, int]:
            create_search_index_state(session=s, desired_generation=0, active_generation=0)
            job = create_job(session=s, status=JobStatus.RUNNING)
            url = create_ingest_url(session=s, job=job)
            s.flush()
            return job.id, url.id

        job_id, item_id = await run_sync_seed(seed)
        facts = facts_for(png_bytes, phash="0000000000000000")
        images = FakeImages()
        images.set(rule.staging_key(item_id), facts)
        env = FakeEnv(db=db, http=image_http(png_bytes), image_facts=images)
        acts = IngestActivities(env, poll_interval_s=0.01)

        result = await ActivityEnvironment().run(
            acts.item,
            Input(job_id=job_id, item_id=item_id, source=RemoteUrl(url=REMOTE_URL)),
        )
        assert result.outcome == "processed"
        async with db.read_session() as session:
            job = await session.get(Job, job_id)
            assert job.progress == 100.0

    async def test_item_honours_cancellation(
        self, db: SavepointDb, run_sync_seed, png_bytes: bytes
    ) -> None:
        def seed(s: Session) -> tuple[str, int]:
            create_search_index_state(session=s, desired_generation=0, active_generation=0)
            job = create_job(session=s, status=JobStatus.RUNNING)
            url = create_ingest_url(session=s, job=job)
            s.flush()
            return job.id, url.id

        job_id, item_id = await run_sync_seed(seed)
        facts = facts_for(png_bytes, phash="0000000000000000")
        images = FakeImages()
        images.set(rule.staging_key(item_id), facts)

        blocked = asyncio.Event()

        class Blocking(FakeInference):
            async def annotate(self, input, *, progress=None):
                await blocked.wait()
                return await super().annotate(input, progress=progress)

        env = FakeEnv(db=db, http=image_http(png_bytes), image_facts=images, inference=Blocking())
        acts = IngestActivities(env, poll_interval_s=0.01)

        activity_env = ActivityEnvironment()
        activity_env.cancel()
        with pytest.raises(asyncio.CancelledError):
            await activity_env.run(
                acts.item,
                Input(job_id=job_id, item_id=item_id, source=RemoteUrl(url=REMOTE_URL)),
            )


class TestFinish:
    async def test_finish_sweeps_and_completes(self, db: SavepointDb, run_sync_seed) -> None:
        def seed(s: Session) -> str:
            job = create_job(session=s, status=JobStatus.RUNNING)
            create_ingest_url(session=s, job=job, status=ProcessingStatus.DONE)
            create_ingest_url(session=s, job=job, status=ProcessingStatus.PENDING)
            return job.id

        job_id = await run_sync_seed(seed)
        env = FakeEnv(db=db)
        acts = IngestActivities(env)
        result = await ActivityEnvironment().run(acts.finish, Finish(job_id=job_id))
        assert result.processed == 1 and result.failed == 1
        async with db.read_session() as session:
            job = await session.get(Job, job_id)
            assert job.status is JobStatus.FAILED  # one item failed -> job failed
