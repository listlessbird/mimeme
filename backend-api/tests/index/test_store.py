from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session
from tests.factories import (
    create_image,
    create_job,
    create_processing,
    create_search_index_state,
)
from tests.job.conftest import SavepointDb
from tests.support.storage import Memory

from mimeme import index, search, storage
from mimeme.config import Settings
from mimeme.db.schema import JobStatus, JobType, ProcessingStatus
from mimeme.index import ops, rule
from mimeme.index.store import Store
from mimeme.job import ops as job_ops


class _Activation:
    def __init__(self) -> None:
        self.serving: str | None = None
        self.candidate: str | None = None

    async def load(self, generation: search.Load) -> search.Loaded:
        self.candidate = generation.version
        return search.Loaded(
            version=generation.version,
            embed_model="test/embed",
            dimension=2,
            image_count=1,
            faiss_version="1.13.2",
            onnxruntime_version="1.27.0",
            encoder_revision="rev",
        )

    async def switch(self, version: str) -> search.Status:
        assert self.candidate == version
        self.serving = version
        self.candidate = None
        return await self.status()

    async def rollback(self, failed_version: str) -> search.Status:
        if self.serving == failed_version:
            self.serving = None
        return await self.status()

    async def status(self) -> search.Status:
        return search.Status(ready=self.serving is not None, serving_version=self.serving)

    async def clear(self) -> search.Status:
        self.serving = None
        self.candidate = None
        return await self.status()


async def test_prepare_claims_and_freezes_object_reference_snapshot_atomically(
    index_db: SavepointDb, run_sync_seed
) -> None:
    def seed(session: Session) -> str:
        job = create_job(session=session, type=JobType.REBUILD_INDEX)
        image = create_image(session=session)
        processing = create_processing(session=session, image=image)
        processing.embed_status = ProcessingStatus.DONE
        processing.embed_model = "test/embed"
        processing.embed_dim = 2
        processing.embed_s3_key = "embeddings/one.npy"
        processing.embed_text_present = True
        session.flush()
        create_search_index_state(session=session, desired_generation=4, active_generation=1)
        return job.id

    job_id = await run_sync_seed(seed)
    prepared = await ops.prepare(
        index_db,
        Settings(),
        index.PrepareInput(
            job_id=job_id,
            workflow_id=rule.workflow_id(job_id),
            force=False,
            trigger=index.Trigger.MANUAL,
            model="test/embed",
            index_type="flat",
        ),
    )

    assert prepared.build is not None
    assert prepared.build.target_generation == 4
    assert prepared.build.embeddings == [
        index.Embedding(
            image_id=prepared.build.embeddings[0].image_id,
            image_key="embeddings/one.npy",
            text_key="embeddings/one_text.npy",
        )
    ]
    assert (await job_ops.find_exn(index_db, job_id)).status is JobStatus.RUNNING


async def test_prepare_retry_resumes_its_own_claim(index_db: SavepointDb, run_sync_seed) -> None:
    def seed(session: Session) -> str:
        job = create_job(session=session, type=JobType.REBUILD_INDEX)
        image = create_image(session=session)
        processing = create_processing(session=session, image=image)
        processing.embed_status = ProcessingStatus.DONE
        processing.embed_model = "test/embed"
        processing.embed_dim = 2
        processing.embed_s3_key = "embeddings/retry.npy"
        session.flush()
        create_search_index_state(session=session, desired_generation=2, active_generation=1)
        return job.id

    job_id = await run_sync_seed(seed)
    request = index.PrepareInput(
        job_id=job_id,
        workflow_id=rule.workflow_id(job_id),
        trigger=index.Trigger.MANUAL,
        model="test/embed",
        index_type="flat",
    )

    first = await ops.prepare(index_db, Settings(), request)
    second = await ops.prepare(index_db, Settings(), request)

    assert first.decision == second.decision == "build"
    assert second.build is not None and second.build.target_generation == 2


async def test_startup_reconciliation_clears_compute_when_db_has_no_active_version(
    index_db: SavepointDb,
) -> None:
    remote = _Activation()
    remote.serving = "orphaned"

    status = await ops.reconcile(index_db, Memory(), remote)

    assert status is not None
    assert status.serving_version is None


async def test_activation_commit_is_atomic_and_releases_the_claim(
    index_db: SavepointDb, run_sync_seed
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
            rebuild_claimed_at=datetime.now(UTC),
        )
        return job.id

    job_id = await run_sync_seed(seed)
    manifest = index.Manifest(
        version="v2-g5-test",
        target_generation=5,
        model="test/embed",
        index_type="flat",
        encoder=index.Encoder(repo="encoder", revision="rev", variant="model.onnx"),
        dimension=2,
        image_count=1,
        files=[
            index.File(
                name=name,
                key=f"indexes/v2-g5-test/{name}",
                sha256="0" * 64,
                length=1,
            )
            for name in ("index.faiss", "mapping.json", "metadata.json")
        ],
        complete_key="indexes/v2-g5-test/complete.json",
    )

    async with index_db.write_session() as session:
        await Store(session).activate(job_id=job_id, manifest=manifest)

    status = await job_ops.index_status(index_db)
    assert status.active_version == manifest.version
    assert status.view.active_generation == 5
    assert status.view.active_claim is None
    assert (await job_ops.find_exn(index_db, job_id)).status is JobStatus.COMPLETED


async def test_activation_retry_confirms_existing_db_and_compute_state(
    index_db: SavepointDb, run_sync_seed
) -> None:
    def seed(session: Session) -> str:
        job = create_job(session=session, type=JobType.REBUILD_INDEX, status=JobStatus.RUNNING)
        session.flush()
        create_search_index_state(
            session=session,
            desired_generation=3,
            active_generation=1,
            rebuild_job_id=job.id,
            rebuild_target_generation=3,
            rebuild_claimed_at=datetime.now(UTC),
        )
        return job.id

    job_id = await run_sync_seed(seed)
    artifacts = Memory()
    digest = storage.Checksum.of(b"data").value
    manifest = index.Manifest(
        version="v2-g3-retry",
        target_generation=3,
        model="test/embed",
        index_type="flat",
        encoder=index.Encoder(repo="encoder", revision="rev", variant="model.onnx"),
        dimension=2,
        image_count=1,
        files=[
            index.File(
                name=name,
                key=f"indexes/v2-g3-retry/{name}",
                sha256=digest,
                length=4,
            )
            for name in ("index.faiss", "mapping.json", "metadata.json")
        ],
        complete_key="indexes/v2-g3-retry/complete.json",
    )
    for file in manifest.files:
        await artifacts.put_bytes(storage.Object(file.key), b"data", content_type="x")
    await artifacts.put_bytes(
        storage.Object(manifest.complete_key),
        manifest.model_dump_json().encode(),
        content_type="application/json",
    )
    input = index.ActivateInput(
        job_id=job_id,
        target_generation=3,
        result=index.Result(outcome="built", manifest=manifest),
    )
    remote = _Activation()

    first = await ops.activate(index_db, artifacts, remote, input)
    second = await ops.activate(index_db, artifacts, remote, input)

    assert first.version == second.version == manifest.version
    assert remote.serving == manifest.version
    assert (await job_ops.find_exn(index_db, job_id)).status is JobStatus.COMPLETED
