"""Idempotency tests for workflows and activities.

Verifies that:
- Activities are safe to retry after partial completion
- Duplicate workflow starts are handled correctly
- State mutations are idempotent (upsert, not duplicate-insert)
"""

from __future__ import annotations

import uuid

import pytest
from temporalio import activity
from temporalio.client import WorkflowExecutionStatus, WorkflowFailureError
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Worker

from activities.workflow_state.activities import (
    complete_ingest_job_activity,
    mark_ingest_url_done_activity,
    mark_ingest_url_failed_activity,
    save_annotations_activity,
    save_embedding_info_activity,
)
from activities.workflow_state.models import (
    CompleteIngestJobInput,
    MarkIngestUrlDoneInput,
    MarkIngestUrlFailedInput,
    SaveAnnotationsInput,
    SaveEmbeddingInfoInput,
)
from shared.models.orm import Annotation, IngestURL, Job, JobStatus, Processing, ProcessingStatus
from tests.factories import ImageFactory, IngestURLFactory, JobFactory, ProcessingFactory
from workflows.ingest import IngestWorkflow
from workflows.models import IngestWorkflowInput

# ---------------------------------------------------------------------------
# Activity-level idempotency (DB mutations safe to retry)
# ---------------------------------------------------------------------------


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


@pytest.mark.usefixtures("_patch_session_scope")
class TestMarkDoneIdempotency:
    async def test_calling_mark_done_twice_is_safe(self, db_session, activity_env) -> None:
        """mark_ingest_url_done should be safe to call twice for the same URL."""
        job = JobFactory(session=db_session)
        image = ImageFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        inp = MarkIngestUrlDoneInput(ingest_url_id=url.id, image_id=image.id)

        await activity_env.run(mark_ingest_url_done_activity, inp)
        db_session.refresh(url)
        assert url.status == ProcessingStatus.DONE
        assert url.image_id == image.id

        # Call again — should be idempotent, not crash
        await activity_env.run(mark_ingest_url_done_activity, inp)
        db_session.refresh(url)
        assert url.status == ProcessingStatus.DONE
        assert url.image_id == image.id


@pytest.mark.usefixtures("_patch_session_scope")
class TestMarkFailedIdempotency:
    async def test_calling_mark_failed_twice_is_safe(self, db_session, activity_env) -> None:
        job = JobFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        inp = MarkIngestUrlFailedInput(ingest_url_id=url.id, error="First error")

        await activity_env.run(mark_ingest_url_failed_activity, inp)
        await activity_env.run(mark_ingest_url_failed_activity, inp)

        db_session.refresh(url)
        assert url.status == ProcessingStatus.FAILED

    async def test_mark_failed_after_done_overwrites(self, db_session, activity_env) -> None:
        """If a URL was already DONE and then marked failed (weird edge case),
        it should update to FAILED without error."""
        job = JobFactory(session=db_session)
        image = ImageFactory(session=db_session)
        url = IngestURLFactory(session=db_session, job=job)
        db_session.flush()

        # First mark as done
        done_inp = MarkIngestUrlDoneInput(ingest_url_id=url.id, image_id=image.id)
        await activity_env.run(mark_ingest_url_done_activity, done_inp)
        db_session.refresh(url)
        assert url.status == ProcessingStatus.DONE

        # Then mark as failed
        fail_inp = MarkIngestUrlFailedInput(ingest_url_id=url.id, error="Late failure")
        await activity_env.run(mark_ingest_url_failed_activity, fail_inp)
        db_session.refresh(url)
        assert url.status == ProcessingStatus.FAILED


@pytest.mark.usefixtures("_patch_session_scope")
class TestSaveAnnotationsIdempotency:
    async def test_double_save_upserts(self, db_session, activity_env) -> None:
        """save_annotations uses get-or-create pattern, so retries should upsert."""
        image = ImageFactory(session=db_session)
        ProcessingFactory(session=db_session, image=image)
        db_session.flush()

        inp = SaveAnnotationsInput(
            image_id=image.id,
            caption="A cat meme",
            caption_model="moondream2",
            ocr_text="LOL",
            ocr_model="moondream2",
        )
        await activity_env.run(save_annotations_activity, inp)
        await activity_env.run(save_annotations_activity, inp)

        # Should have exactly one annotation row
        count = db_session.query(Annotation).filter_by(image_id=image.id).count()
        assert count == 1

    async def test_retry_with_different_values_updates(self, db_session, activity_env) -> None:
        """A retry with updated values should overwrite, not create a duplicate."""
        image = ImageFactory(session=db_session)
        ProcessingFactory(session=db_session, image=image)
        db_session.flush()

        inp1 = SaveAnnotationsInput(
            image_id=image.id,
            caption="v1 caption",
            caption_model="model-v1",
            ocr_text="v1 ocr",
            ocr_model="model-v1",
        )
        await activity_env.run(save_annotations_activity, inp1)

        inp2 = SaveAnnotationsInput(
            image_id=image.id,
            caption="v2 caption",
            caption_model="model-v2",
            ocr_text="v2 ocr",
            ocr_model="model-v2",
        )
        await activity_env.run(save_annotations_activity, inp2)

        ann = db_session.query(Annotation).filter_by(image_id=image.id).first()
        assert ann.caption_text == "v2 caption"
        assert ann.ocr_text == "v2 ocr"


@pytest.mark.usefixtures("_patch_session_scope")
class TestSaveEmbeddingInfoIdempotency:
    async def test_double_save_is_safe(self, db_session, activity_env) -> None:
        """save_embedding_info on the same Processing row should update, not crash."""
        image = ImageFactory(session=db_session)
        proc = ProcessingFactory(session=db_session, image=image)
        db_session.flush()

        inp = SaveEmbeddingInfoInput(
            image_id=image.id,
            model="siglip2-base",
            dimension=768,
            image_embedding_key="embeddings/abc.npy",
        )
        await activity_env.run(save_embedding_info_activity, inp)
        await activity_env.run(save_embedding_info_activity, inp)

        db_session.refresh(proc)
        assert proc.embed_status == ProcessingStatus.DONE
        assert proc.embed_dim == 768


@pytest.mark.usefixtures("_patch_session_scope")
class TestCompleteJobIdempotency:
    async def test_double_complete_is_safe(self, db_session, activity_env) -> None:
        """Completing a job twice should overwrite result, not crash."""
        job = JobFactory(session=db_session, status=JobStatus.RUNNING)
        db_session.flush()

        inp = CompleteIngestJobInput(
            job_id=job.id, processed=5, failed=0, duplicates=1
        )
        await activity_env.run(complete_ingest_job_activity, inp)

        db_session.refresh(job)
        assert job.status == JobStatus.COMPLETED
        first_completed_at = job.completed_at

        # Call again
        await activity_env.run(complete_ingest_job_activity, inp)
        db_session.refresh(job)
        assert job.status == JobStatus.COMPLETED
        # completed_at may be updated — that's fine


# ---------------------------------------------------------------------------
# Workflow-level idempotency (duplicate workflow starts)
# ---------------------------------------------------------------------------


# Minimal mocks for workflow-level tests
@activity.defn(name="ingest_initialize_activity")
async def _mock_init(job_id: str) -> dict:
    from activities.workflow_state.models import IngestInitOutput, IngestUrlItem

    return IngestInitOutput(urls=[]).model_dump()


@activity.defn(name="complete_ingest_job_activity")
async def _mock_complete(input: dict) -> None:
    pass


@activity.defn(name="update_job_progress_activity")
async def _mock_progress(input: dict) -> None:
    pass


class TestWorkflowDuplicateStart:
    async def test_same_workflow_id_rejects_second_start(self) -> None:
        """Starting a workflow with the same ID twice should raise."""
        task_queue = str(uuid.uuid4())
        workflow_id = f"ingest-test-{uuid.uuid4().hex[:8]}"

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[IngestWorkflow],
                activities=[_mock_init, _mock_complete, _mock_progress],
            ):
                # First start succeeds
                await env.client.execute_workflow(
                    IngestWorkflow.run,
                    IngestWorkflowInput(job_id="test-1"),
                    id=workflow_id,
                    task_queue=task_queue,
                )

                # Second start with SAME workflow ID should fail
                with pytest.raises(Exception):
                    await env.client.start_workflow(
                        IngestWorkflow.run,
                        IngestWorkflowInput(job_id="test-1"),
                        id=workflow_id,
                        task_queue=task_queue,
                    )
