"""Integration tests for RebuildIndexWorkflow using Temporal's test environment."""

from __future__ import annotations

import uuid

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.indexing.models import (
    BuildIndexInput,
    BuildIndexOutput,
    GarbageCollectOutput,
    SwapIndexInput,
)
from activities.workflow_state.models import (
    CompleteRebuildJobInput,
    FailRebuildJobInput,
    StartRebuildJobInput,
    UpdateJobProgressInput,
)
from workflows.models import RebuildIndexWorkflowInput
from workflows.rebuild_index import RETRY_INDEX_BUILD, RebuildIndexWorkflow

# ---------------------------------------------------------------------------
# Mock activities
# ---------------------------------------------------------------------------

_activity_calls: list[str] = []
_progress_values: list[float] = []


@activity.defn(name="start_rebuild_job_activity")
async def mock_start_rebuild(input: StartRebuildJobInput) -> None:
    _activity_calls.append("start_rebuild_job_activity")


@activity.defn(name="update_job_progress_activity")
async def mock_update_progress(input: UpdateJobProgressInput) -> None:
    _activity_calls.append("update_job_progress_activity")
    _progress_values.append(input.progress)


@activity.defn(name="build_index_activity")
async def mock_build_index(input: BuildIndexInput) -> BuildIndexOutput:
    _activity_calls.append("build_index_activity")
    return BuildIndexOutput(
        version="v-test-001",
        num_vectors=500,
        dimension=768,
        s3_key="indexes/v-test-001/index.faiss",
        text_num_vectors=500,
        text_s3_key="indexes/v-test-001/text_index.faiss",
    )


@activity.defn(name="swap_index_activity")
async def mock_swap_index(input: SwapIndexInput) -> None:
    _activity_calls.append("swap_index_activity")


@activity.defn(name="garbage_collect_indexes_activity")
async def mock_gc_indexes() -> GarbageCollectOutput:
    _activity_calls.append("garbage_collect_indexes_activity")
    return GarbageCollectOutput(removed_versions=["v-old-001"])


@activity.defn(name="complete_rebuild_job_activity")
async def mock_complete_rebuild(input: CompleteRebuildJobInput) -> None:
    _activity_calls.append("complete_rebuild_job_activity")


@activity.defn(name="fail_rebuild_job_activity")
async def mock_fail_rebuild(input: FailRebuildJobInput) -> None:
    _activity_calls.append("fail_rebuild_job_activity")


ALL_MOCK_ACTIVITIES = [
    mock_start_rebuild,
    mock_update_progress,
    mock_build_index,
    mock_swap_index,
    mock_gc_indexes,
    mock_complete_rebuild,
    mock_fail_rebuild,
]


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    _activity_calls.clear()
    _progress_values.clear()


def test_index_build_retry_policy_is_single_attempt() -> None:
    assert isinstance(RETRY_INDEX_BUILD, RetryPolicy)
    assert RETRY_INDEX_BUILD.maximum_attempts == 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRebuildWorkflowHappyPath:
    async def test_full_rebuild_succeeds(self) -> None:
        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[RebuildIndexWorkflow],
                activities=ALL_MOCK_ACTIVITIES,
            ):
                result = await env.client.execute_workflow(
                    RebuildIndexWorkflow.run,
                    RebuildIndexWorkflowInput(
                        job_id="rebuild-1", model_name="test-model", index_type="flat"
                    ),
                    id=str(uuid.uuid4()),
                    task_queue=task_queue,
                )

        assert result.job_id == "rebuild-1"
        assert result.version == "v-test-001"
        assert result.num_vectors == 500
        assert result.dimension == 768
        assert result.removed_versions == ["v-old-001"]

    async def test_activity_call_order(self) -> None:
        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[RebuildIndexWorkflow],
                activities=ALL_MOCK_ACTIVITIES,
            ):
                await env.client.execute_workflow(
                    RebuildIndexWorkflow.run,
                    RebuildIndexWorkflowInput(
                        job_id="rebuild-order", model_name="test-model", index_type="flat"
                    ),
                    id=str(uuid.uuid4()),
                    task_queue=task_queue,
                )

        assert _activity_calls == [
            "start_rebuild_job_activity",
            "update_job_progress_activity",  # 10%
            "build_index_activity",
            "update_job_progress_activity",  # 70%
            "swap_index_activity",
            "update_job_progress_activity",  # 90%
            "garbage_collect_indexes_activity",
            "complete_rebuild_job_activity",
        ]

    async def test_progress_values(self) -> None:
        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[RebuildIndexWorkflow],
                activities=ALL_MOCK_ACTIVITIES,
            ):
                await env.client.execute_workflow(
                    RebuildIndexWorkflow.run,
                    RebuildIndexWorkflowInput(
                        job_id="rebuild-progress", model_name="test-model", index_type="flat"
                    ),
                    id=str(uuid.uuid4()),
                    task_queue=task_queue,
                )

        assert _progress_values == [10.0, 70.0, 90.0]


class TestRebuildWorkflowFailure:
    async def test_build_index_failure_marks_job_failed(self) -> None:
        @activity.defn(name="build_index_activity")
        async def mock_build_fail(input: BuildIndexInput) -> BuildIndexOutput:
            _activity_calls.append("build_index_activity")
            raise ApplicationError("No embeddings found", non_retryable=True)

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_build_index"]
        activities.append(mock_build_fail)

        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[RebuildIndexWorkflow],
                activities=activities,
            ):
                with pytest.raises(WorkflowFailureError):
                    await env.client.execute_workflow(
                        RebuildIndexWorkflow.run,
                        RebuildIndexWorkflowInput(
                            job_id="rebuild-fail", model_name="test-model", index_type="flat"
                        ),
                        id=str(uuid.uuid4()),
                        task_queue=task_queue,
                    )

        assert "fail_rebuild_job_activity" in _activity_calls
        assert "complete_rebuild_job_activity" not in _activity_calls

    async def test_swap_failure_marks_job_failed(self) -> None:
        @activity.defn(name="swap_index_activity")
        async def mock_swap_fail(input: SwapIndexInput) -> None:
            _activity_calls.append("swap_index_activity")
            raise ApplicationError("Failed to swap index", non_retryable=True)

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_swap_index"]
        activities.append(mock_swap_fail)

        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[RebuildIndexWorkflow],
                activities=activities,
            ):
                with pytest.raises(WorkflowFailureError):
                    await env.client.execute_workflow(
                        RebuildIndexWorkflow.run,
                        RebuildIndexWorkflowInput(
                            job_id="rebuild-swap-fail", model_name="test-model", index_type="flat"
                        ),
                        id=str(uuid.uuid4()),
                        task_queue=task_queue,
                    )

        assert "fail_rebuild_job_activity" in _activity_calls


class TestRebuildWorkflowGarbageCollect:
    async def test_empty_removed_versions(self) -> None:
        @activity.defn(name="garbage_collect_indexes_activity")
        async def mock_gc_empty() -> GarbageCollectOutput:
            _activity_calls.append("garbage_collect_indexes_activity")
            return GarbageCollectOutput(removed_versions=[])

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_gc_indexes"]
        activities.append(mock_gc_empty)

        task_queue = str(uuid.uuid4())
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[RebuildIndexWorkflow],
                activities=activities,
            ):
                result = await env.client.execute_workflow(
                    RebuildIndexWorkflow.run,
                    RebuildIndexWorkflowInput(
                        job_id="rebuild-no-gc", model_name="test-model", index_type="flat"
                    ),
                    id=str(uuid.uuid4()),
                    task_queue=task_queue,
                )

        assert result.removed_versions == []
