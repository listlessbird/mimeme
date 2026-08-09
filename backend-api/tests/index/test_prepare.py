from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session
from tests.factories import (
    create_image,
    create_job,
    create_processing,
    create_search_index_state,
)
from tests.job.conftest import SavepointDb
from tests.support.storage import Memory

from mimeme import index
from mimeme.config import Settings
from mimeme.db.schema import JobStatus, JobType, ProcessingStatus
from mimeme.index import ops, rule
from mimeme.job import ops as job_ops


def _settings(**overrides: object) -> Settings:
    settings = Settings()
    return settings.model_copy(update={"index": settings.index.model_copy(update=overrides)})


def _request(job_id: str | None, **overrides: object) -> index.PrepareInput:
    fields: dict[str, object] = {
        "job_id": job_id,
        "workflow_id": rule.workflow_id(job_id or "scheduled"),
        "trigger": index.Trigger.MANUAL,
        "model": "test/embed",
        "index_type": "flat",
    }
    fields.update(overrides)
    return index.PrepareInput(**fields)  # type: ignore[arg-type]


def _embedded(session: Session, *, key: str, text_present: bool | None) -> None:
    image = create_image(session=session)
    processing = create_processing(session=session, image=image)
    processing.embed_status = ProcessingStatus.DONE
    processing.embed_model = "test/embed"
    processing.embed_dim = 2
    processing.embed_s3_key = key
    processing.embed_text_present = text_present
    session.flush()


class TestTextPresence:
    async def test_absent_and_unresolved_rows_carry_no_text_reference(
        self, index_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, type=JobType.REBUILD_INDEX)
            _embedded(session, key="embeddings/present.npy", text_present=True)
            _embedded(session, key="embeddings/absent.npy", text_present=False)
            _embedded(session, key="embeddings/unresolved.npy", text_present=None)
            create_search_index_state(session=session, desired_generation=2, active_generation=1)
            return job.id

        job_id = await run_sync_seed(seed)
        store = Memory()
        prepared = await ops.prepare(index_db, store, Settings(), _request(job_id))

        assert prepared.build is not None
        build = await ops.load_build(store, prepared.build)
        by_key = {item.image_key: item.text_key for item in build.embeddings}
        assert by_key == {
            "embeddings/present.npy": "embeddings/present_text.npy",
            "embeddings/absent.npy": None,
            "embeddings/unresolved.npy": None,
        }

    async def test_planned_reads_counts_one_object_per_referenced_vector(
        self, index_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, type=JobType.REBUILD_INDEX)
            _embedded(session, key="embeddings/a.npy", text_present=True)
            _embedded(session, key="embeddings/b.npy", text_present=False)
            create_search_index_state(session=session, desired_generation=2, active_generation=1)
            return job.id

        job_id = await run_sync_seed(seed)
        prepared = await ops.prepare(index_db, Memory(), Settings(), _request(job_id))

        assert prepared.build is not None
        assert prepared.build.planned_reads == 3


class TestSettleWindow:
    async def test_a_moving_dirty_stream_defers_the_rebuild(
        self, index_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, type=JobType.REBUILD_INDEX)
            _embedded(session, key="embeddings/a.npy", text_present=False)
            create_search_index_state(
                session=session,
                desired_generation=9,
                active_generation=1,
                last_dirty_at=datetime.now(UTC) - timedelta(minutes=1),
                last_reconciled_at=datetime.now(UTC) - timedelta(minutes=30),
            )
            return job.id

        job_id = await run_sync_seed(seed)
        prepared = await ops.prepare(index_db, Memory(), Settings(), _request(job_id))

        assert prepared.decision == "deferred"
        assert (await job_ops.index_status(index_db)).view.active_claim is None
        assert (await job_ops.find_exn(index_db, job_id)).status is JobStatus.COMPLETED

    async def test_a_settled_dirty_stream_rebuilds(
        self, index_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, type=JobType.REBUILD_INDEX)
            _embedded(session, key="embeddings/a.npy", text_present=False)
            create_search_index_state(
                session=session,
                desired_generation=9,
                active_generation=1,
                last_dirty_at=datetime.now(UTC) - timedelta(minutes=30),
                last_reconciled_at=datetime.now(UTC) - timedelta(minutes=40),
            )
            return job.id

        job_id = await run_sync_seed(seed)
        prepared = await ops.prepare(index_db, Memory(), Settings(), _request(job_id))

        assert prepared.decision == "build"

    async def test_maximum_staleness_escapes_a_never_settling_stream(
        self, index_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, type=JobType.REBUILD_INDEX)
            _embedded(session, key="embeddings/a.npy", text_present=False)
            create_search_index_state(
                session=session,
                desired_generation=9,
                active_generation=1,
                last_dirty_at=datetime.now(UTC) - timedelta(seconds=5),
                last_reconciled_at=datetime.now(UTC) - timedelta(hours=7),
            )
            return job.id

        job_id = await run_sync_seed(seed)
        prepared = await ops.prepare(index_db, Memory(), Settings(), _request(job_id))

        assert prepared.decision == "build"

    async def test_forced_manual_rebuild_ignores_the_settle_window(
        self, index_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, type=JobType.REBUILD_INDEX)
            _embedded(session, key="embeddings/a.npy", text_present=False)
            create_search_index_state(
                session=session,
                desired_generation=9,
                active_generation=1,
                last_dirty_at=datetime.now(UTC),
                last_reconciled_at=datetime.now(UTC),
            )
            return job.id

        job_id = await run_sync_seed(seed)
        prepared = await ops.prepare(index_db, Memory(), Settings(), _request(job_id, force=True))

        assert prepared.decision == "build"

    async def test_a_continuous_dirty_stream_defers_every_scheduled_run(
        self, index_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> None:
            _embedded(session, key="embeddings/a.npy", text_present=False)
            create_search_index_state(
                session=session,
                desired_generation=1,
                active_generation=0,
                last_reconciled_at=datetime.now(UTC),
            )

        await run_sync_seed(seed)
        decisions = []
        for tick in range(5):
            async with index_db.write_session() as session:
                await job_ops.Store(session).mark_dirty(reason="embedding_saved")
            prepared = await ops.prepare(
                index_db,
                Memory(),
                Settings(),
                _request(f"scheduled-{tick}", trigger=index.Trigger.SCHEDULED),
            )
            decisions.append(prepared.decision)

        assert decisions == ["deferred"] * 5

    def test_settle_predicate_treats_a_never_reconciled_index_as_overdue(self) -> None:
        now = datetime.now(UTC)
        assert rule.settled(
            now=now,
            last_dirty_at=now,
            last_reconciled_at=None,
            settle=timedelta(minutes=10),
            max_stale=timedelta(hours=6),
        )


class TestReadCount:
    async def test_a_large_planned_read_count_reports_without_refusing(
        self, index_db: SavepointDb, run_sync_seed
    ) -> None:
        def seed(session: Session) -> str:
            job = create_job(session=session, type=JobType.REBUILD_INDEX)
            for position in range(3):
                _embedded(session, key=f"embeddings/{position}.npy", text_present=True)
            create_search_index_state(session=session, desired_generation=2, active_generation=1)
            return job.id

        job_id = await run_sync_seed(seed)
        prepared = await ops.prepare(index_db, Memory(), Settings(), _request(job_id))

        assert prepared.decision == "build"
        assert prepared.build is not None
        assert prepared.build.planned_reads == 6
