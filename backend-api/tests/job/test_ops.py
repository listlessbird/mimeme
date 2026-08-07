from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from mimeme.db.schema import (
    Annotation,
    IngestURL,
    Job,
    JobStatus,
    JobType,
    Processing,
    ProcessingStatus,
    RebuildTrigger,
    SearchIndexState,
)
from mimeme.ingest.model import RemoteUrl as RemoteImageUrlInput
from mimeme.ingest.model import Staged as StagedUploadInput
from mimeme.job import ops
from mimeme.job.model import (
    ClaimOwnership,
    ClaimTarget,
    InvalidState,
    NotFound,
    SourceIngestItem,
    StateMissing,
)
from tests.factories import (
    create_image,
    create_index_build,
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_processing,
    create_search_index_state,
    create_source_item,
    create_source_run,
)
from tests.job.conftest import PoolDb, SavepointDb


class TestCreate:
    async def test_create_ingest_dedups_and_inserts(
        self, job_db: SavepointDb, async_db_connection
    ) -> None:
        creation = await ops.create_ingest(
            job_db,
            inputs=[
                RemoteImageUrlInput(url="https://a/1.jpg"),
                RemoteImageUrlInput(url="https://a/1.jpg"),
                StagedUploadInput(artifact_key="staged/x.jpg"),
            ],
            dataset="d",
            tags=["t"],
            callback_url="https://cb",
        )
        assert creation.queued == 2
        assert creation.duplicates == 1
        assert creation.dataset == "d"

        view = await ops.find_exn(job_db, creation.job_id)
        assert view.type is JobType.INGEST
        assert view.status is JobStatus.PENDING

    async def test_create_source_ingest_records_source_refs(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> tuple[int, int, int]:
            source = create_ingestion_source(session=session)
            run = create_source_run(session=session, source=source)
            item = create_source_item(session=session, source=source)
            session.flush()
            return source.id, run.id, item.id

        source_id, run_id, item_id = await run_sync_seed(seed)
        creation = await ops.create_source_ingest(
            job_db,
            dataset="feed",
            items=[
                SourceIngestItem(
                    url="https://a/1.jpg",
                    source_id=source_id,
                    source_run_id=run_id,
                    source_item_id=item_id,
                )
            ],
        )
        assert creation.queued == 1
        async with job_db.read_session() as session:
            row = (
                await session.scalars(select(IngestURL).where(IngestURL.job_id == creation.job_id))
            ).first()
        assert row is not None
        assert row.source_item_id == item_id

    async def test_create_rebuild_returns_view(self, job_db: SavepointDb) -> None:
        creation = await ops.create_rebuild(job_db, force=True, model_name="m", index_type="flat")
        assert creation.force is True
        assert creation.job.type is JobType.REBUILD_INDEX


class TestReads:
    async def test_find_missing_is_none(self, job_db: SavepointDb) -> None:
        assert await ops.find(job_db, "nope") is None

    async def test_find_exn_missing_raises(self, job_db: SavepointDb) -> None:
        with pytest.raises(NotFound):
            await ops.find_exn(job_db, "nope")

    async def test_list_empty(self, job_db: SavepointDb) -> None:
        page = await ops.list_jobs(job_db, status=None, job_type=None, limit=20)
        assert page.jobs == [] and page.total == 0

    async def test_list_filters_by_status_and_type(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> None:
            create_job(session=session, type=JobType.INGEST, status=JobStatus.PENDING)
            create_job(session=session, type=JobType.REBUILD_INDEX, status=JobStatus.COMPLETED)

        await run_sync_seed(seed)

        by_status = await ops.list_jobs(job_db, status=JobStatus.PENDING, job_type=None, limit=20)
        assert by_status.total == 1
        by_type = await ops.list_jobs(job_db, status=None, job_type=JobType.REBUILD_INDEX, limit=20)
        assert by_type.total == 1

    async def test_get_job_parses_invalid_json_as_raw(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, status=JobStatus.COMPLETED)
            job.result = "not-json"
            return job.id

        job_id = await run_sync_seed(seed)
        view = await ops.find_exn(job_db, job_id)
        assert view.result is not None and view.result.raw == "not-json"  # type: ignore[union-attr]


class TestCancellation:
    async def test_record_workflow_id(self, job_db: SavepointDb, run_sync_seed) -> None:
        job_id = await run_sync_seed(lambda s: create_job(session=s).id)
        await ops.record_workflow_id(job_db, job_id, "wf-9")
        async with job_db.read_session() as session:
            assert (await session.get(Job, job_id)).workflow_id == "wf-9"

    async def test_cancel_pending_returns_workflow(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        job_id = await run_sync_seed(
            lambda s: create_job(session=s, status=JobStatus.RUNNING, workflow_id="wf-1").id
        )
        cancellation = await ops.request_cancellation(job_db, job_id)
        assert cancellation.workflow_id == "wf-1"
        await ops.mark_cancelled(job_db, job_id)
        assert (await ops.find_exn(job_db, job_id)).status is JobStatus.CANCELLED

    @pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.FAILED])
    async def test_cancel_terminal_raises(
        self, job_db: SavepointDb, run_sync_seed, status: JobStatus
    ) -> None:
        job_id = await run_sync_seed(lambda s: create_job(session=s, status=status).id)
        with pytest.raises(InvalidState):
            await ops.request_cancellation(job_db, job_id)

    async def test_cancel_missing_raises(self, job_db: SavepointDb) -> None:
        with pytest.raises(NotFound):
            await ops.request_cancellation(job_db, "nope")

    async def test_release_claim_swallows_foreign_and_missing(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        await ops.release_claim(job_db, "no-state")  # StateMissing swallowed

        def seed(session: Session) -> tuple[str, str]:
            owner = create_job(session=session, type=JobType.REBUILD_INDEX)
            other = create_job(session=session, type=JobType.REBUILD_INDEX)
            session.flush()
            create_search_index_state(
                session=session,
                desired_generation=5,
                active_generation=1,
                rebuild_job_id=owner.id,
                rebuild_target_generation=5,
                rebuild_claimed_at=datetime.now(UTC),
            )
            return owner.id, other.id

        owner_id, other_id = await run_sync_seed(seed)
        await ops.release_claim(job_db, other_id)  # ownership error swallowed, claim intact
        async with job_db.read_session() as session:
            assert (await session.get(SearchIndexState, 1)).rebuild_job_id == owner_id


class TestTransitions:
    async def test_start_marks_running(self, job_db: SavepointDb, run_sync_seed) -> None:
        job_id = await run_sync_seed(lambda s: create_job(session=s).id)
        await ops.start(job_db, job_id)
        view = await ops.find_exn(job_db, job_id)
        assert view.status is JobStatus.RUNNING and view.started_at is not None

    @pytest.mark.parametrize(
        ("failed", "expected"),
        [(0, JobStatus.COMPLETED), (2, JobStatus.FAILED)],
    )
    async def test_complete_ingest_status(
        self, job_db: SavepointDb, run_sync_seed, failed: int, expected: JobStatus
    ) -> None:
        job_id = await run_sync_seed(lambda s: create_job(session=s, status=JobStatus.RUNNING).id)
        assert await ops.complete_ingest(
            job_db, job_id=job_id, processed=3, failed=failed, duplicates=1
        )
        view = await ops.find_exn(job_db, job_id)
        assert view.status is expected
        assert view.progress == 100.0
        assert view.result is not None

    async def test_complete_ingest_missing_returns_false(self, job_db: SavepointDb) -> None:
        assert not await ops.complete_ingest(
            job_db, job_id="nope", processed=0, failed=0, duplicates=0
        )

    async def test_fail_rebuild_truncates(self, job_db: SavepointDb, run_sync_seed) -> None:
        job_id = await run_sync_seed(
            lambda s: create_job(session=s, type=JobType.REBUILD_INDEX, status=JobStatus.RUNNING).id
        )
        assert await ops.fail_rebuild(job_db, job_id, "x" * 5000)
        view = await ops.find_exn(job_db, job_id)
        assert view.status is JobStatus.FAILED
        assert view.message is not None and len(view.message) == 2000

    async def test_complete_rebuild_missing_raises(self, job_db: SavepointDb) -> None:
        with pytest.raises(NotFound):
            await ops.complete_rebuild(
                job_db,
                job_id="nope",
                version="v",
                num_vectors=1,
                dimension=1,
                removed_versions=[],
                text_num_vectors=None,
            )

    async def test_progress_preserves_message_when_absent(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        job_id = await run_sync_seed(lambda s: create_job(session=s, message="old").id)
        assert await ops.progress(job_db, job_id, 42.0)
        view = await ops.find_exn(job_db, job_id)
        assert view.progress == 42.0 and view.message == "old"


class TestIngestItems:
    async def test_initialize_marks_running_and_returns_urls(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, type=JobType.INGEST)
            create_ingest_url(session=session, job=job)
            create_ingest_url(session=session, job=job)
            return job.id

        job_id = await run_sync_seed(seed)
        init = await ops.initialize_ingest(job_db, job_id)
        assert len(init.urls) == 2
        assert (await ops.find_exn(job_db, job_id)).status is JobStatus.RUNNING

    async def test_record_stage_missing_is_false(self, job_db: SavepointDb) -> None:
        from mimeme.db.schema import IngestStage

        assert not await ops.record_stage(job_db, 999999, IngestStage.DOWNLOADING)

    async def test_mark_item_done_with_existing_image(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> tuple[int, int]:
            job = create_job(session=session)
            image = create_image(session=session)
            url = create_ingest_url(session=session, job=job)
            session.flush()
            return url.id, image.id

        url_id, image_id = await run_sync_seed(seed)
        done = await ops.mark_item_done(job_db, url_id, image_id)
        assert done.found and done.image_exists
        async with job_db.read_session() as session:
            row = await session.get(IngestURL, url_id)
            assert row.status is ProcessingStatus.DONE and row.image_id == image_id

    async def test_mark_item_done_missing_image_fails_url(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> int:
            job = create_job(session=session)
            url = create_ingest_url(session=session, job=job)
            session.flush()
            return url.id

        url_id = await run_sync_seed(seed)
        done = await ops.mark_item_done(job_db, url_id, 999999)
        assert done.found and not done.image_exists
        async with job_db.read_session() as session:
            assert (await session.get(IngestURL, url_id)).status is ProcessingStatus.FAILED

    async def test_mark_item_done_missing_url(self, job_db: SavepointDb) -> None:
        done = await ops.mark_item_done(job_db, 999999, 1)
        assert not done.found and done.image_exists is None

    async def test_mark_item_failed_truncates(self, job_db: SavepointDb, run_sync_seed) -> None:
        def seed(session: Session) -> int:
            job = create_job(session=session)
            url = create_ingest_url(session=session, job=job)
            session.flush()
            return url.id

        url_id = await run_sync_seed(seed)
        assert await ops.mark_item_failed(job_db, url_id, "x" * 3000)
        async with job_db.read_session() as session:
            row = await session.get(IngestURL, url_id)
            assert row.status is ProcessingStatus.FAILED and len(row.error_message) == 1000


class TestInference:
    async def test_save_annotations_upserts(self, job_db: SavepointDb, run_sync_seed) -> None:
        def seed(session: Session) -> int:
            image = create_image(session=session)
            create_processing(session=session, image=image)
            return image.id

        image_id = await run_sync_seed(seed)
        assert await ops.save_annotations(
            job_db,
            image_id=image_id,
            caption="cap",
            caption_model="m",
            ocr_text="ocr",
            ocr_model="m",
        )
        assert await ops.save_annotations(
            job_db,
            image_id=image_id,
            caption="cap2",
            caption_model="m2",
            ocr_text="ocr2",
            ocr_model="m2",
        )
        async with job_db.read_session() as session:
            rows = (
                await session.scalars(select(Annotation).where(Annotation.image_id == image_id))
            ).all()
        assert len(rows) == 1 and rows[0].caption_text == "cap2"

    async def test_save_embedding_marks_dirty_once(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> int:
            create_search_index_state(session=session, desired_generation=1, active_generation=1)
            image = create_image(session=session)
            create_processing(session=session, image=image)
            return image.id

        image_id = await run_sync_seed(seed)
        first = await ops.save_embedding(
            job_db,
            image_id=image_id,
            model="m",
            dimension=768,
            image_embedding_key="k",
            text_embedding_key="k_text",
        )
        assert first.index_changed and first.desired_generation == 2
        # identical retry does not re-increment
        second = await ops.save_embedding(
            job_db,
            image_id=image_id,
            model="m",
            dimension=768,
            image_embedding_key="k",
            text_embedding_key="k_text",
        )
        assert not second.index_changed
        assert (await ops.index_status(job_db)).view.desired_generation == 2

    async def test_re_embedding_drops_the_recorded_shard_position(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> int:
            create_search_index_state(session=session, desired_generation=1, active_generation=1)
            image = create_image(session=session)
            processing = create_processing(session=session, image=image)
            processing.embed_model = "m"
            processing.embed_s3_key = "k"
            processing.embed_shard = 4
            processing.embed_row = 7
            session.flush()
            return image.id

        image_id = await run_sync_seed(seed)
        await ops.save_embedding(
            job_db,
            image_id=image_id,
            model="m",
            dimension=768,
            image_embedding_key="k2",
            text_embedding_key=None,
        )

        async with job_db.read_session() as session:
            row = (
                await session.scalars(select(Processing).where(Processing.image_id == image_id))
            ).one()
        assert (row.embed_shard, row.embed_row) == (None, None)

    async def test_an_unchanged_embedding_keeps_its_shard_position(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> int:
            create_search_index_state(session=session, desired_generation=1, active_generation=1)
            image = create_image(session=session)
            processing = create_processing(session=session, image=image)
            processing.embed_model = "m"
            processing.embed_s3_key = "k"
            processing.embed_shard = 4
            processing.embed_row = 7
            session.flush()
            return image.id

        image_id = await run_sync_seed(seed)
        await ops.save_embedding(
            job_db,
            image_id=image_id,
            model="m",
            dimension=768,
            image_embedding_key="k",
            text_embedding_key=None,
        )

        async with job_db.read_session() as session:
            row = (
                await session.scalars(select(Processing).where(Processing.image_id == image_id))
            ).one()
        assert (row.embed_shard, row.embed_row) == (4, 7)

    async def test_save_embedding_missing_processing_does_not_increment(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> int:
            create_search_index_state(session=session, desired_generation=3, active_generation=3)
            image = create_image(session=session)
            return image.id

        image_id = await run_sync_seed(seed)
        saved = await ops.save_embedding(
            job_db,
            image_id=image_id,
            model="m",
            dimension=768,
            image_embedding_key="k",
            text_embedding_key="k_text",
        )
        assert not saved.found
        assert (await ops.index_status(job_db)).view.desired_generation == 3


class TestFreshnessAndClaims:
    async def test_index_status_reports_active_version(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> None:
            create_search_index_state(session=session, desired_generation=3, active_generation=1)
            create_index_build(session=session, version="v-cur", is_active=True)

        await run_sync_seed(seed)
        status = await ops.index_status(job_db)
        assert status.view.is_stale and status.active_version == "v-cur"

    async def test_prepare_manual_dirty_claims_build(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            create_search_index_state(session=session, desired_generation=4, active_generation=1)
            return create_job(session=session, type=JobType.REBUILD_INDEX).id

        job_id = await run_sync_seed(seed)
        decision = await ops.prepare_rebuild(
            job_db,
            job_id=job_id,
            workflow_id="wf",
            force=False,
            trigger=RebuildTrigger.MANUAL,
            now=datetime.now(UTC),
            claim_timeout=timedelta(minutes=180),
        )
        assert decision.decision == "build"
        assert decision.job_id == job_id
        assert decision.target_generation == 4
        async with job_db.read_session() as session:
            assert (await session.get(SearchIndexState, 1)).rebuild_job_id == job_id

    async def test_prepare_missing_state_raises(self, job_db: SavepointDb) -> None:
        with pytest.raises(StateMissing):
            await ops.prepare_rebuild(
                job_db,
                job_id=None,
                workflow_id="wf",
                force=False,
                trigger=RebuildTrigger.SCHEDULED,
                now=datetime.now(UTC),
                claim_timeout=timedelta(minutes=180),
            )

    async def test_prepare_scheduled_creates_job(self, job_db: SavepointDb, run_sync_seed) -> None:
        await run_sync_seed(
            lambda s: create_search_index_state(
                session=s, desired_generation=4, active_generation=1
            )
        )
        decision = await ops.prepare_rebuild(
            job_db,
            job_id=None,
            workflow_id="wf-sched",
            force=False,
            trigger=RebuildTrigger.SCHEDULED,
            now=datetime.now(UTC),
            claim_timeout=timedelta(minutes=180),
        )
        assert decision.decision == "build" and decision.job_id is not None
        async with job_db.read_session() as session:
            job = await session.get(Job, decision.job_id)
            assert job is not None and job.workflow_id == "wf-sched"

    async def test_prepare_clean_manual_completes_skip(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            create_search_index_state(session=session, desired_generation=2, active_generation=2)
            create_index_build(
                session=session, version="v-live", is_active=True, num_vectors=7, dimension=768
            )
            return create_job(session=session, type=JobType.REBUILD_INDEX).id

        job_id = await run_sync_seed(seed)
        decision = await ops.prepare_rebuild(
            job_db,
            job_id=job_id,
            workflow_id="wf",
            force=False,
            trigger=RebuildTrigger.MANUAL,
            now=datetime.now(UTC),
            claim_timeout=timedelta(minutes=180),
        )
        assert decision.decision == "clean"
        view = await ops.find_exn(job_db, job_id)
        assert view.status is JobStatus.COMPLETED
        assert view.result is not None and view.result.skipped is True  # type: ignore[union-attr]

    async def test_prepare_busy_when_live_claim_held(
        self, job_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            owner = create_job(
                session=session, type=JobType.REBUILD_INDEX, status=JobStatus.RUNNING
            )
            session.flush()
            create_search_index_state(
                session=session,
                desired_generation=5,
                active_generation=1,
                rebuild_job_id=owner.id,
                rebuild_target_generation=5,
                rebuild_claimed_at=datetime.now(UTC),
            )
            return create_job(session=session, type=JobType.REBUILD_INDEX).id

        new_job = await run_sync_seed(seed)
        decision = await ops.prepare_rebuild(
            job_db,
            job_id=new_job,
            workflow_id="wf",
            force=False,
            trigger=RebuildTrigger.MANUAL,
            now=datetime.now(UTC),
            claim_timeout=timedelta(minutes=180),
        )
        assert decision.decision == "busy"

    async def test_prepare_reclaims_expired_claim(self, job_db: SavepointDb, run_sync_seed) -> None:
        claimed = datetime.now(UTC) - timedelta(hours=4)

        def seed(session: Session) -> tuple[str, str]:
            owner = create_job(
                session=session, type=JobType.REBUILD_INDEX, status=JobStatus.RUNNING
            )
            session.flush()
            create_search_index_state(
                session=session,
                desired_generation=5,
                active_generation=1,
                rebuild_job_id=owner.id,
                rebuild_target_generation=5,
                rebuild_claimed_at=claimed,
            )
            new = create_job(session=session, type=JobType.REBUILD_INDEX)
            return owner.id, new.id

        owner_id, new_job = await run_sync_seed(seed)
        decision = await ops.prepare_rebuild(
            job_db,
            job_id=new_job,
            workflow_id="wf",
            force=False,
            trigger=RebuildTrigger.MANUAL,
            now=datetime.now(UTC),
            claim_timeout=timedelta(minutes=180),
        )
        assert decision.decision == "build" and decision.job_id == new_job
        assert (await ops.find_exn(job_db, owner_id)).status is JobStatus.FAILED

    async def test_activate_advances_generation(self, job_db: SavepointDb, run_sync_seed) -> None:
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
        view = await ops.activate_generation(
            job_db, job_id=job_id, target_generation=5, reconciled_at=datetime.now(UTC)
        )
        assert view.active_generation == 5

    async def test_activate_wrong_owner_raises(self, job_db: SavepointDb, run_sync_seed) -> None:
        def seed(session: Session) -> str:
            owner = create_job(session=session, type=JobType.REBUILD_INDEX)
            other = create_job(session=session, type=JobType.REBUILD_INDEX)
            session.flush()
            create_search_index_state(
                session=session,
                desired_generation=5,
                active_generation=1,
                rebuild_job_id=owner.id,
                rebuild_target_generation=5,
                rebuild_claimed_at=datetime.now(UTC),
            )
            return other.id

        other_id = await run_sync_seed(seed)
        with pytest.raises(ClaimOwnership):
            await ops.activate_generation(
                job_db, job_id=other_id, target_generation=5, reconciled_at=datetime.now(UTC)
            )

    async def test_activate_wrong_target_raises(self, job_db: SavepointDb, run_sync_seed) -> None:
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
        with pytest.raises(ClaimTarget):
            await ops.activate_generation(
                job_db, job_id=job_id, target_generation=4, reconciled_at=datetime.now(UTC)
            )


class TestConcurrency:
    async def test_concurrent_prepare_only_one_claims(self, pool_db: PoolDb) -> None:
        job_a = f"rebuild-{'a' * 8}"
        job_b = f"rebuild-{'b' * 8}"
        try:
            async with pool_db.write_session() as session:
                session.add(Job(id=job_a, type=JobType.REBUILD_INDEX))
                session.add(Job(id=job_b, type=JobType.REBUILD_INDEX))
                await session.flush()
                await session.execute(delete(SearchIndexState))
                session.add(SearchIndexState(id=1, desired_generation=4, active_generation=1))

            now = datetime.now(UTC)

            async def run(job_id: str) -> str:
                decision = await ops.prepare_rebuild(
                    pool_db,
                    job_id=job_id,
                    workflow_id="wf",
                    force=False,
                    trigger=RebuildTrigger.MANUAL,
                    now=now,
                    claim_timeout=timedelta(minutes=180),
                )
                return decision.decision

            first, second = await asyncio.gather(run(job_a), run(job_b))
            assert {first, second} == {"build", "busy"}
        finally:
            async with pool_db.write_session() as session:
                await session.execute(delete(SearchIndexState))
                await session.execute(delete(Job).where(Job.id.in_([job_a, job_b])))
