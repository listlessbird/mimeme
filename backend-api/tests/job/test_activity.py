from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session
from temporalio.testing import ActivityEnvironment

from mimeme.db.schema import JobType, RebuildTrigger, SearchIndexState
from mimeme.job import (
    REBUILD_STATE,
    JobActivities,
    PrepareCommand,
    ReconcileCommand,
    ReleaseCommand,
    StartCommand,
)
from tests.factories import create_job, create_search_index_state
from tests.job.conftest import SavepointDb


@pytest.fixture()
def acts(job_db: SavepointDb) -> JobActivities:
    return JobActivities(job_db, rebuild_claim_timeout=timedelta(minutes=180))


@pytest.fixture()
def env() -> ActivityEnvironment:
    return ActivityEnvironment()


def test_rebuild_state_name_is_temporary() -> None:
    assert REBUILD_STATE == "mimeme.job.rebuild-state.tmp"


class TestRebuildState:
    async def test_prepare_builds(
        self, acts: JobActivities, env: ActivityEnvironment, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            create_search_index_state(session=session, desired_generation=4, active_generation=1)
            return create_job(session=session, type=JobType.REBUILD_INDEX).id

        job_id = await run_sync_seed(seed)
        out = await env.run(
            acts.rebuild_state,
            PrepareCommand(
                job_id=job_id, workflow_id="wf", force=False, trigger=RebuildTrigger.MANUAL
            ),
        )
        assert out.decision is not None and out.decision.decision == "build"

    async def test_start_reconcile_release_cycle(
        self, acts: JobActivities, env: ActivityEnvironment, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, type=JobType.REBUILD_INDEX)
            session.flush()
            create_search_index_state(
                session=session,
                desired_generation=5,
                active_generation=1,
                rebuild_job_id=job.id,
                rebuild_target_generation=5,
                rebuild_claimed_at=datetime.now(UTC),
            )
            return job.id

        job_id = await run_sync_seed(seed)
        await env.run(acts.rebuild_state, StartCommand(job_id=job_id))
        await env.run(acts.rebuild_state, ReconcileCommand(job_id=job_id, target_generation=5))
        released = await env.run(acts.rebuild_state, ReleaseCommand(job_id=job_id))
        assert released.released is True
        async with job_db.read_session() as session:
            state = await session.get(SearchIndexState, 1)
            assert state.active_generation == 5 and state.rebuild_job_id is None
