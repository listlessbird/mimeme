from __future__ import annotations

from temporalio.testing import ActivityEnvironment

from mimeme.db.schema import ProcessingStatus, SourceRunStatus
from mimeme.source.activity import SourceActivities
from mimeme.source.model import DiscoverInput, FinishInput
from tests.factories import (
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_run,
)
from tests.job.conftest import SavepointDb
from tests.source.conftest import FakeEnv, FakeHttp, meme_response

MEME_URL = "https://meme-api.com/gimme/memes/50"


class TestDiscoverActivity:
    async def test_wrapper_heartbeats_and_returns_result(
        self, db: SavepointDb, run_sync_seed
    ) -> None:
        source_id = await run_sync_seed(
            lambda s: create_ingestion_source(
                session=s, adapter_config={"subreddits": ["memes"]}, max_items_per_run=50
            ).id
        )
        http = FakeHttp()
        http.set(MEME_URL, meme_response("aaa"))
        activities = SourceActivities(FakeEnv(db=db, source_http=http))

        beats: list[tuple] = []
        env = ActivityEnvironment()
        env.on_heartbeat = lambda *args: beats.append(args)

        result = await env.run(activities.discover, DiscoverInput(source_id=source_id))

        assert result.queued == 1 and result.ingest_job_id is not None
        assert beats  # heartbeat fired at least once during fetching


class TestFinishActivity:
    async def test_wrapper_returns_accounting(self, db: SavepointDb, run_sync_seed) -> None:
        def seed(s) -> int:
            source = create_ingestion_source(session=s)
            run = create_source_run(session=s, source=source, status=SourceRunStatus.RUNNING)
            job = create_job(session=s)
            create_ingest_url(
                session=s, job=job, source_run_id=run.id, status=ProcessingStatus.DONE
            )
            return run.id

        run_id = await run_sync_seed(seed)
        activities = SourceActivities(FakeEnv(db=db))

        result = await ActivityEnvironment().run(
            activities.finish, FinishInput(source_run_id=run_id)
        )
        assert result.status == SourceRunStatus.COMPLETED and result.queued == 1
