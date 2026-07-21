from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session
from temporalio.testing import ActivityEnvironment

from mimeme.db.schema import (
    IngestStage,
    IngestURL,
    Job,
    JobStatus,
    JobType,
    ProcessingStatus,
    RebuildTrigger,
    SearchIndexState,
)
from mimeme.job import (
    INGEST_STATE,
    REBUILD_STATE,
    AnnotationSave,
    CompleteIngestCommand,
    CompleteItemCommand,
    EmbeddingSave,
    FailItemCommand,
    InitializeCommand,
    JobActivities,
    PrepareCommand,
    ProgressCommand,
    ReconcileCommand,
    ReleaseCommand,
    SaveInferenceCommand,
    StageCommand,
    StartCommand,
)
from tests.factories import (
    create_image,
    create_ingest_url,
    create_job,
    create_processing,
    create_search_index_state,
)
from tests.job.conftest import SavepointDb


@pytest.fixture()
def acts(job_db: SavepointDb) -> JobActivities:
    return JobActivities(job_db, rebuild_claim_timeout=timedelta(minutes=180))


@pytest.fixture()
def env() -> ActivityEnvironment:
    return ActivityEnvironment()


def test_activity_names_are_temporary() -> None:
    assert INGEST_STATE == "mimeme.job.ingest-state.tmp"
    assert REBUILD_STATE == "mimeme.job.rebuild-state.tmp"


class TestIngestState:
    async def test_initialize_returns_urls(
        self, acts: JobActivities, env: ActivityEnvironment, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, type=JobType.INGEST)
            create_ingest_url(session=session, job=job)
            return job.id

        job_id = await run_sync_seed(seed)
        out = await env.run(acts.ingest_state, InitializeCommand(job_id=job_id))
        assert out.init is not None and len(out.init.urls) == 1

    async def test_stage_and_complete_item(
        self, acts: JobActivities, env: ActivityEnvironment, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> tuple[int, int]:
            job = create_job(session=session)
            image = create_image(session=session)
            url = create_ingest_url(session=session, job=job)
            session.flush()
            return url.id, image.id

        url_id, image_id = await run_sync_seed(seed)
        await env.run(
            acts.ingest_state, StageCommand(ingest_url_id=url_id, stage=IngestStage.EMBEDDING)
        )
        out = await env.run(
            acts.ingest_state, CompleteItemCommand(ingest_url_id=url_id, image_id=image_id)
        )
        assert out.found and out.image_exists
        async with job_db.read_session() as session:
            assert (await session.get(IngestURL, url_id)).status is ProcessingStatus.DONE

    async def test_fail_item_is_idempotent(
        self, acts: JobActivities, env: ActivityEnvironment, run_sync_seed
    ) -> None:
        def seed(session: Session) -> int:
            job = create_job(session=session)
            url = create_ingest_url(session=session, job=job)
            session.flush()
            return url.id

        url_id = await run_sync_seed(seed)
        first = await env.run(
            acts.ingest_state, FailItemCommand(ingest_url_id=url_id, error="boom")
        )
        second = await env.run(
            acts.ingest_state, FailItemCommand(ingest_url_id=url_id, error="boom")
        )
        assert first.found and second.found

    async def test_save_inference_persists_both(
        self, acts: JobActivities, env: ActivityEnvironment, run_sync_seed
    ) -> None:
        def seed(session: Session) -> int:
            create_search_index_state(session=session, desired_generation=1, active_generation=1)
            image = create_image(session=session)
            create_processing(session=session, image=image)
            return image.id

        image_id = await run_sync_seed(seed)
        out = await env.run(
            acts.ingest_state,
            SaveInferenceCommand(
                annotation=AnnotationSave(
                    image_id=image_id, caption="c", caption_model="m", ocr_text="o", ocr_model="m"
                ),
                embedding=EmbeddingSave(
                    image_id=image_id, model="m", dimension=768, image_embedding_key="k"
                ),
            ),
        )
        assert out.index_changed and out.desired_generation == 2

    async def test_progress_and_complete(
        self, acts: JobActivities, env: ActivityEnvironment, job_db: SavepointDb, run_sync_seed
    ) -> None:
        job_id = await run_sync_seed(lambda s: create_job(session=s, status=JobStatus.RUNNING).id)
        await env.run(acts.ingest_state, ProgressCommand(job_id=job_id, progress=50.0))
        await env.run(
            acts.ingest_state,
            CompleteIngestCommand(job_id=job_id, processed=5, failed=0, duplicates=1),
        )
        async with job_db.read_session() as session:
            job = await session.get(Job, job_id)
            assert job.status is JobStatus.COMPLETED and job.progress == 100.0


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
