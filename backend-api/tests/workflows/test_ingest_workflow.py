"""Integration tests for IngestWorkflow using Temporal's test environment.

Uses WorkflowEnvironment.start_time_skipping() with mocked activities
to test the workflow orchestration logic without real DB/S3/GPU.
"""

from __future__ import annotations

import uuid

import pytest
from activities.embedding.models import EmbedBatchInput, EmbedBatchOutput, EmbedImageOutput
from activities.storage.models import (
    DownloadImageInput,
    DownloadImageOutput,
    ProcessImageInput,
    ProcessImageOutput,
)
from activities.vision.models import AnnotateImageInput, AnnotateImageOutput
from activities.workflow_state.models import (
    CompleteIngestJobInput,
    IngestInitOutput,
    IngestUrlItem,
    MarkIngestUrlDoneInput,
    MarkIngestUrlFailedInput,
    SaveAnnotationsInput,
    SaveEmbeddingInfoInput,
    UpdateJobProgressInput,
)
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from workflows.ingest import IngestWorkflow
from workflows.models import IngestWorkflowInput

# ---------------------------------------------------------------------------
# Mock activities — each returns a canned output
# ---------------------------------------------------------------------------

_activity_calls: list[str] = []


@activity.defn(name="ingest_initialize_activity")
async def mock_ingest_initialize(job_id: str) -> IngestInitOutput:
    _activity_calls.append("ingest_initialize_activity")
    return IngestInitOutput(
        urls=[
            IngestUrlItem(id=1, url="https://example.com/img1.jpg"),
            IngestUrlItem(id=2, url="https://example.com/img2.png"),
        ]
    )


@activity.defn(name="download_image_activity")
async def mock_download_image(input: DownloadImageInput) -> DownloadImageOutput:
    _activity_calls.append("download_image_activity")
    return DownloadImageOutput(
        ingest_url_id=input.ingest_url_id,
        local_path="/tmp/test.jpg",
        filename="test.jpg",
        success=True,
    )


@activity.defn(name="process_image_activity")
async def mock_process_image(input: ProcessImageInput) -> ProcessImageOutput:
    _activity_calls.append("process_image_activity")
    return ProcessImageOutput(
        ingest_url_id=input.ingest_url_id,
        image_id=input.ingest_url_id * 100,
        sha256=f"hash-{input.ingest_url_id}",
        s3_key=f"images/test/{input.ingest_url_id}.jpg",
        width=800,
        height=600,
        format="jpeg",
        is_duplicate=False,
    )


@activity.defn(name="annotate_image_activity")
async def mock_annotate_image(input: AnnotateImageInput) -> AnnotateImageOutput:
    _activity_calls.append("annotate_image_activity")
    return AnnotateImageOutput(
        image_id=input.image_id,
        caption="A funny meme",
        caption_model="moondream2",
        ocr_text="IMPACT TEXT",
        ocr_model="moondream2",
    )


@activity.defn(name="save_annotations_activity")
async def mock_save_annotations(input: SaveAnnotationsInput) -> None:
    _activity_calls.append("save_annotations_activity")


@activity.defn(name="embed_batch_activity")
async def mock_embed_batch(input: EmbedBatchInput) -> EmbedBatchOutput:
    _activity_calls.append("embed_batch_activity")
    results = [
        EmbedImageOutput(
            image_id=item.image_id,
            image_embedding_key=f"embeddings/{item.image_id}.npy",
            text_embedding_key=f"embeddings/{item.image_id}_text.npy",
            model="siglip2-base",
            dimension=768,
        )
        for item in input.items
    ]
    return EmbedBatchOutput(results=results, failed_ids=[])


@activity.defn(name="save_embedding_info_activity")
async def mock_save_embedding_info(input: SaveEmbeddingInfoInput) -> None:
    _activity_calls.append("save_embedding_info_activity")


@activity.defn(name="mark_ingest_url_done_activity")
async def mock_mark_done(input: MarkIngestUrlDoneInput) -> None:
    _activity_calls.append("mark_ingest_url_done_activity")


@activity.defn(name="mark_ingest_url_failed_activity")
async def mock_mark_failed(input: MarkIngestUrlFailedInput) -> None:
    _activity_calls.append("mark_ingest_url_failed_activity")


@activity.defn(name="update_job_progress_activity")
async def mock_update_progress(input: UpdateJobProgressInput) -> None:
    _activity_calls.append("update_job_progress_activity")


@activity.defn(name="complete_ingest_job_activity")
async def mock_complete_job(input: CompleteIngestJobInput) -> None:
    _activity_calls.append("complete_ingest_job_activity")


ALL_MOCK_ACTIVITIES = [
    mock_ingest_initialize,
    mock_download_image,
    mock_process_image,
    mock_annotate_image,
    mock_save_annotations,
    mock_embed_batch,
    mock_save_embedding_info,
    mock_mark_done,
    mock_mark_failed,
    mock_update_progress,
    mock_complete_job,
]


@pytest.fixture(autouse=True)
def _reset_activity_calls() -> None:
    _activity_calls.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestWorkflowHappyPath:
    async def test_all_urls_succeed(self) -> None:
        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[IngestWorkflow],
                activities=ALL_MOCK_ACTIVITIES,
            ):
                result = await env.client.execute_workflow(
                    IngestWorkflow.run,
                    IngestWorkflowInput(job_id="test-job-1"),
                    id=str(uuid.uuid4()),
                    task_queue=task_queue,
                )

        assert result.job_id == "test-job-1"
        assert result.total == 2
        assert result.processed == 2
        assert result.failed == 0
        assert result.duplicates == 0

        assert "ingest_initialize_activity" in _activity_calls
        assert "complete_ingest_job_activity" in _activity_calls
        assert _activity_calls.count("download_image_activity") == 2
        assert _activity_calls.count("mark_ingest_url_done_activity") == 2


class TestIngestWorkflowPartialFailure:
    async def test_download_failure_marks_url_failed(self) -> None:
        """When download returns success=False, the URL is marked failed."""

        @activity.defn(name="download_image_activity")
        async def mock_download_fail_first(
            input: DownloadImageInput,
        ) -> DownloadImageOutput:
            _activity_calls.append("download_image_activity")
            if input.ingest_url_id == 1:
                return DownloadImageOutput(
                    ingest_url_id=1,
                    local_path="",
                    filename="",
                    success=False,
                    error="HTTP 404",
                )
            return DownloadImageOutput(
                ingest_url_id=input.ingest_url_id,
                local_path="/tmp/test.jpg",
                filename="test.jpg",
                success=True,
            )

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_download_image"]
        activities.append(mock_download_fail_first)

        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[IngestWorkflow],
                activities=activities,
            ):
                result = await env.client.execute_workflow(
                    IngestWorkflow.run,
                    IngestWorkflowInput(job_id="test-partial"),
                    id=str(uuid.uuid4()),
                    task_queue=task_queue,
                )

        assert result.processed == 1
        assert result.failed == 1
        assert "mark_ingest_url_failed_activity" in _activity_calls


class TestIngestWorkflowDuplicateHandling:
    async def test_duplicate_image_skips_processing(self) -> None:
        """When process_image returns is_duplicate=True, remaining steps are skipped."""

        @activity.defn(name="process_image_activity")
        async def mock_process_duplicate(
            input: ProcessImageInput,
        ) -> ProcessImageOutput:
            _activity_calls.append("process_image_activity")
            return ProcessImageOutput(
                ingest_url_id=input.ingest_url_id,
                image_id=42,
                sha256="dup-hash",
                s3_key="images/test/dup.jpg",
                width=800,
                height=600,
                format="jpeg",
                is_duplicate=True,
            )

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_process_image"]
        activities.append(mock_process_duplicate)

        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[IngestWorkflow],
                activities=activities,
            ):
                result = await env.client.execute_workflow(
                    IngestWorkflow.run,
                    IngestWorkflowInput(job_id="test-dup"),
                    id=str(uuid.uuid4()),
                    task_queue=task_queue,
                )

        assert result.duplicates == 2
        assert result.processed == 0
        # Annotation/embed should NOT be called for duplicates
        assert "annotate_image_activity" not in _activity_calls
        assert "embed_batch_activity" not in _activity_calls


class TestIngestWorkflowEmptyUrls:
    async def test_empty_url_list(self) -> None:
        """When the job has no URLs, the workflow completes immediately."""

        @activity.defn(name="ingest_initialize_activity")
        async def mock_init_empty(job_id: str) -> IngestInitOutput:
            _activity_calls.append("ingest_initialize_activity")
            return IngestInitOutput(urls=[])

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_ingest_initialize"]
        activities.append(mock_init_empty)

        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[IngestWorkflow],
                activities=activities,
            ):
                result = await env.client.execute_workflow(
                    IngestWorkflow.run,
                    IngestWorkflowInput(job_id="test-empty"),
                    id=str(uuid.uuid4()),
                    task_queue=task_queue,
                )

        assert result.total == 0
        assert result.processed == 0
        assert result.failed == 0

    async def test_all_urls_fail(self) -> None:
        """When every URL fails download, workflow completes with all failures."""

        @activity.defn(name="download_image_activity")
        async def mock_download_all_fail(
            input: DownloadImageInput,
        ) -> DownloadImageOutput:
            _activity_calls.append("download_image_activity")
            return DownloadImageOutput(
                ingest_url_id=input.ingest_url_id,
                local_path="",
                filename="",
                success=False,
                error="Server error",
            )

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_download_image"]
        activities.append(mock_download_all_fail)

        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[IngestWorkflow],
                activities=activities,
            ):
                result = await env.client.execute_workflow(
                    IngestWorkflow.run,
                    IngestWorkflowInput(job_id="test-all-fail"),
                    id=str(uuid.uuid4()),
                    task_queue=task_queue,
                )

        assert result.total == 2
        assert result.processed == 0
        assert result.failed == 2


class TestIngestWorkflowActivityException:
    async def test_embed_throws_url_marked_failed(self) -> None:
        """If embed_batch raises, the URL is caught and marked failed."""

        @activity.defn(name="embed_batch_activity")
        async def mock_embed_fail(input: EmbedBatchInput) -> EmbedBatchOutput:
            _activity_calls.append("embed_batch_activity")
            raise ApplicationError("GPU out of memory", non_retryable=True)

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_embed_batch"]
        activities.append(mock_embed_fail)

        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[IngestWorkflow],
                activities=activities,
            ):
                result = await env.client.execute_workflow(
                    IngestWorkflow.run,
                    IngestWorkflowInput(job_id="test-embed-fail"),
                    id=str(uuid.uuid4()),
                    task_queue=task_queue,
                )

        assert result.failed == 2
        assert "mark_ingest_url_failed_activity" in _activity_calls


class TestIngestWorkflowIdempotency:
    async def test_same_workflow_id_rejects_second_start(self) -> None:
        """Starting a workflow with the same ID while one is running should
        raise WorkflowAlreadyStartedError.

        We use a slow activity (sleeps via heartbeat loop) so the first
        workflow is still running when we attempt the second start.
        """
        import asyncio
        import concurrent.futures
        import threading

        from temporalio.exceptions import WorkflowAlreadyStartedError

        started = threading.Event()
        barrier = threading.Event()

        @activity.defn(name="ingest_initialize_activity")
        def mock_init_blocking(job_id: str) -> IngestInitOutput:
            """Sync activity that blocks until the test signals it."""
            _activity_calls.append("ingest_initialize_activity")
            started.set()
            barrier.wait(timeout=30)
            return IngestInitOutput(urls=[])

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_ingest_initialize"]
        activities.append(mock_init_blocking)

        task_queue = str(uuid.uuid4())
        workflow_id = f"ingest-test-{uuid.uuid4().hex[:8]}"

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[IngestWorkflow],
                activities=activities,
                activity_executor=concurrent.futures.ThreadPoolExecutor(max_workers=4),
            ):
                # First start — non-blocking handle, workflow stays running
                await env.client.start_workflow(
                    IngestWorkflow.run,
                    IngestWorkflowInput(job_id="test-1"),
                    id=workflow_id,
                    task_queue=task_queue,
                )

                assert await asyncio.to_thread(started.wait, 30)

                try:
                    with pytest.raises(WorkflowAlreadyStartedError):
                        await env.client.start_workflow(
                            IngestWorkflow.run,
                            IngestWorkflowInput(job_id="test-1"),
                            id=workflow_id,
                            task_queue=task_queue,
                        )
                finally:
                    barrier.set()
