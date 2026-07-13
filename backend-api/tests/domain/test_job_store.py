from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tests.factories import create_job

from domain.job_lifecycle import JobLifecycleInvalidStateError, JobLifecycleNotFoundError
from domain.job_rules import IngestJobResultPayload, RawJobResultPayload, RebuildJobResultPayload
from domain.job_store import ApiJobStore
from shared.models import IngestURL, Job, JobStatus, JobType

pytestmark = pytest.mark.usefixtures(
    "_patch_domain_session_scope", "_patch_async_domain_session_scope"
)


async def test_create_ingest_job_deduplicates_urls_and_preserves_order(
    async_db_session: AsyncSession,
) -> None:
    result = await ApiJobStore().create_ingest_job(
        urls=[
            "https://example.com/1.jpg",
            "https://example.com/1.jpg",
            "https://example.com/2.jpg",
        ],
        dataset="memes",
        tags=["funny"],
        callback_url="https://example.com/callback",
    )

    rows = (
        await async_db_session.scalars(
            select(IngestURL).where(IngestURL.job_id == result.job_id).order_by(IngestURL.id)
        )
    ).all()
    assert result.queued == 2
    assert result.duplicates == 1
    assert result.workflow_id == f"ingest-workflow-{result.job_id}"
    assert [row.url for row in rows] == [
        "https://example.com/1.jpg",
        "https://example.com/2.jpg",
    ]


async def test_create_rebuild_job_returns_pending_rebuild_view() -> None:
    result = await ApiJobStore().create_rebuild_job(
        force=True,
        model_name="model-v2",
        index_type="flat",
    )

    assert result.job.type == JobType.REBUILD_INDEX
    assert result.job.status == JobStatus.PENDING
    assert result.workflow_id == f"rebuild-workflow-{result.job.id}"
    assert result.force is True
    assert result.model_name == "model-v2"


async def test_create_ingest_job_records_workflow_id() -> None:
    store = ApiJobStore()
    result = await store.create_ingest_job(
        urls=["https://example.com/1.jpg"],
        dataset=None,
        tags=[],
        callback_url=None,
    )

    await store.record_workflow_id(result.job_id, result.workflow_id)

    assert (await store.get_job(result.job_id)).id == result.job_id
    assert (await store.request_cancellation(result.job_id)).workflow_id == result.workflow_id


async def test_get_job_returns_typed_result_payload(run_sync_seed) -> None:
    def seed(session) -> str:
        job = create_job(session=session, status=JobStatus.COMPLETED)
        job.result = IngestJobResultPayload(processed=5, failed=0, duplicates=0).model_dump_json()
        return job.id

    job_id = await run_sync_seed(seed)

    result = await ApiJobStore().get_job(job_id)

    assert result.result == IngestJobResultPayload(processed=5, failed=0, duplicates=0)


async def test_get_rebuild_job_returns_typed_result_payload(run_sync_seed) -> None:
    def seed(session) -> str:
        job = create_job(session=session, type=JobType.REBUILD_INDEX, status=JobStatus.COMPLETED)
        job.result = RebuildJobResultPayload(
            version="v-1",
            num_vectors=10,
            dimension=768,
            removed_versions=["v-old"],
            text_num_vectors=9,
        ).model_dump_json()
        return job.id

    job_id = await run_sync_seed(seed)

    result = await ApiJobStore().get_job(job_id)

    assert result.result == RebuildJobResultPayload(
        version="v-1",
        num_vectors=10,
        dimension=768,
        removed_versions=["v-old"],
        text_num_vectors=9,
    )


async def test_get_job_preserves_invalid_json_as_raw_result(run_sync_seed) -> None:
    def seed(session) -> str:
        job = create_job(session=session, status=JobStatus.COMPLETED)
        job.result = "not-json"
        return job.id

    job_id = await run_sync_seed(seed)

    result = await ApiJobStore().get_job(job_id)

    assert result.result == RawJobResultPayload(raw="not-json")


async def test_list_jobs_filters_by_status_and_type(run_sync_seed) -> None:
    await run_sync_seed(
        lambda session: (
            create_job(session=session, type=JobType.INGEST, status=JobStatus.PENDING),
            create_job(session=session, type=JobType.REBUILD_INDEX, status=JobStatus.COMPLETED),
        )
    )

    result = await ApiJobStore().list_jobs(
        status=JobStatus.PENDING,
        job_type=JobType.INGEST,
        limit=20,
    )

    assert result.total == 1
    assert result.jobs[0].type == JobType.INGEST
    assert result.jobs[0].status == JobStatus.PENDING


async def test_cancel_pending_job_without_workflow_id(
    async_db_session: AsyncSession, run_sync_seed
) -> None:
    job_id = await run_sync_seed(
        lambda session: create_job(session=session, status=JobStatus.PENDING).id
    )

    store = ApiJobStore()
    cancellation = await store.request_cancellation(job_id)
    await store.mark_cancelled(job_id)

    job = await async_db_session.get(Job, job_id)
    assert job is not None
    assert cancellation.workflow_id is None
    assert job.status == JobStatus.CANCELLED


async def test_cancel_running_job_with_workflow_id(run_sync_seed) -> None:
    job_id = await run_sync_seed(
        lambda session: (
            create_job(session=session, status=JobStatus.RUNNING, workflow_id="wf-123").id
        )
    )

    cancellation = await ApiJobStore().request_cancellation(job_id)

    assert cancellation.workflow_id == "wf-123"


async def test_completed_and_failed_jobs_are_not_cancellable(run_sync_seed) -> None:
    for status in (JobStatus.COMPLETED, JobStatus.FAILED):
        job_id = await run_sync_seed(lambda session: create_job(session=session, status=status).id)

        with pytest.raises(JobLifecycleInvalidStateError):
            await ApiJobStore().request_cancellation(job_id)


async def test_get_missing_job_raises() -> None:
    with pytest.raises(JobLifecycleNotFoundError):
        await ApiJobStore().get_job("missing")
