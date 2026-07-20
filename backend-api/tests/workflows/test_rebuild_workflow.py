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
    PrepareRebuildInput,
    PrepareRebuildOutput,
    ReconcileGenerationInput,
    ReleaseRebuildClaimInput,
    StartRebuildJobInput,
    UpdateJobProgressInput,
)
from shared.models import RebuildTrigger
from workflows.models import RebuildIndexWorkflowInput
from workflows.rebuild_index import RETRY_INDEX_BUILD, RebuildIndexWorkflow

_activity_calls: list[str] = []
_progress_values: list[float] = []
_prepare_results: list[PrepareRebuildOutput] = []
_build_results: list[BuildIndexOutput] = []


def _default_prepare() -> PrepareRebuildOutput:
    return PrepareRebuildOutput(decision="build", job_id="rebuild-1", target_generation=5)


def _default_build() -> BuildIndexOutput:
    return BuildIndexOutput(
        outcome="built",
        version="v-test-001",
        num_vectors=500,
        dimension=768,
        s3_key="indexes/v-test-001/index.faiss",
        text_num_vectors=500,
        text_s3_key="indexes/v-test-001/text_index.faiss",
    )


@activity.defn(name="prepare_rebuild_activity")
async def mock_prepare(input: PrepareRebuildInput) -> PrepareRebuildOutput:
    _activity_calls.append("prepare_rebuild_activity")
    return _prepare_results.pop(0) if _prepare_results else _default_prepare()


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
    return _build_results.pop(0) if _build_results else _default_build()


@activity.defn(name="reconcile_generation_activity")
async def mock_reconcile(input: ReconcileGenerationInput) -> None:
    _activity_calls.append("reconcile_generation_activity")


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


@activity.defn(name="release_rebuild_claim_activity")
async def mock_release_claim(input: ReleaseRebuildClaimInput) -> None:
    _activity_calls.append("release_rebuild_claim_activity")


ALL_MOCK_ACTIVITIES = [
    mock_prepare,
    mock_start_rebuild,
    mock_update_progress,
    mock_build_index,
    mock_reconcile,
    mock_swap_index,
    mock_gc_indexes,
    mock_complete_rebuild,
    mock_fail_rebuild,
    mock_release_claim,
]


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    _activity_calls.clear()
    _progress_values.clear()
    _prepare_results.clear()
    _build_results.clear()


def _manual_input(job_id: str = "rebuild-1") -> RebuildIndexWorkflowInput:
    return RebuildIndexWorkflowInput(
        job_id=job_id,
        model_name="test-model",
        index_type="flat",
        trigger=RebuildTrigger.MANUAL,
    )


def _scheduled_input() -> RebuildIndexWorkflowInput:
    return RebuildIndexWorkflowInput(
        job_id=None,
        model_name="test-model",
        index_type="flat",
        trigger=RebuildTrigger.SCHEDULED,
    )


async def _run(activities: list, input: RebuildIndexWorkflowInput):
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
            return await env.client.execute_workflow(
                RebuildIndexWorkflow.run,
                input,
                id=str(uuid.uuid4()),
                task_queue=task_queue,
            )


def test_index_build_retry_policy_is_single_attempt() -> None:
    assert isinstance(RETRY_INDEX_BUILD, RetryPolicy)
    assert RETRY_INDEX_BUILD.maximum_attempts == 1


class TestRebuildWorkflowHappyPath:
    async def test_full_rebuild_succeeds(self) -> None:
        result = await _run(ALL_MOCK_ACTIVITIES, _manual_input())

        assert result.job_id == "rebuild-1"
        assert result.outcome == "built"
        assert result.version == "v-test-001"
        assert result.num_vectors == 500
        assert result.dimension == 768
        assert result.removed_versions == ["v-old-001"]

    async def test_activity_call_order(self) -> None:
        await _run(ALL_MOCK_ACTIVITIES, _manual_input())

        assert _activity_calls == [
            "prepare_rebuild_activity",
            "start_rebuild_job_activity",
            "update_job_progress_activity",
            "build_index_activity",
            "update_job_progress_activity",
            "swap_index_activity",
            "update_job_progress_activity",
            "garbage_collect_indexes_activity",
            "complete_rebuild_job_activity",
            "release_rebuild_claim_activity",
        ]

    async def test_progress_values(self) -> None:
        await _run(ALL_MOCK_ACTIVITIES, _manual_input())
        assert _progress_values == [10.0, 70.0, 90.0]


class TestRebuildWorkflowDecisions:
    async def test_clean_manual_skips_build(self) -> None:
        _prepare_results.append(
            PrepareRebuildOutput(decision="clean", job_id="rebuild-1", target_generation=None)
        )

        result = await _run(ALL_MOCK_ACTIVITIES, _manual_input())

        assert result.outcome == "skipped"
        assert result.job_id == "rebuild-1"
        assert "build_index_activity" not in _activity_calls
        assert "swap_index_activity" not in _activity_calls

    async def test_clean_scheduled_skips_without_job(self) -> None:
        _prepare_results.append(PrepareRebuildOutput(decision="clean", job_id=None))

        result = await _run(ALL_MOCK_ACTIVITIES, _scheduled_input())

        assert result.outcome == "skipped"
        assert result.job_id is None
        assert _activity_calls == ["prepare_rebuild_activity"]

    async def test_scheduled_busy_exits_without_build(self) -> None:
        _prepare_results.append(PrepareRebuildOutput(decision="busy", job_id=None))

        result = await _run(ALL_MOCK_ACTIVITIES, _scheduled_input())

        assert result.outcome == "busy"
        assert result.job_id is None
        assert _activity_calls == ["prepare_rebuild_activity"]

    async def test_manual_busy_retries_until_claim_is_free(self) -> None:
        _prepare_results.extend(
            [
                PrepareRebuildOutput(decision="busy", job_id="rebuild-1"),
                PrepareRebuildOutput(decision="busy", job_id="rebuild-1"),
            ]
        )

        result = await _run(ALL_MOCK_ACTIVITIES, _manual_input())

        assert result.outcome == "built"
        assert _activity_calls.count("prepare_rebuild_activity") == 3
        assert "build_index_activity" in _activity_calls

    async def test_empty_reconcile_skips_swap(self) -> None:
        _build_results.append(BuildIndexOutput(outcome="empty_reconcile", num_vectors=0))

        result = await _run(ALL_MOCK_ACTIVITIES, _manual_input())

        assert result.outcome == "empty_reconcile"
        assert "reconcile_generation_activity" in _activity_calls
        assert "swap_index_activity" not in _activity_calls
        assert "complete_rebuild_job_activity" in _activity_calls
        assert "release_rebuild_claim_activity" in _activity_calls


class TestRebuildWorkflowFailure:
    async def test_build_index_failure_marks_job_failed_and_releases(self) -> None:
        @activity.defn(name="build_index_activity")
        async def mock_build_fail(input: BuildIndexInput) -> BuildIndexOutput:
            _activity_calls.append("build_index_activity")
            raise ApplicationError("No embeddings found", non_retryable=True)

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_build_index"]
        activities.append(mock_build_fail)

        with pytest.raises(WorkflowFailureError):
            await _run(activities, _manual_input("rebuild-fail"))

        assert "fail_rebuild_job_activity" in _activity_calls
        assert "release_rebuild_claim_activity" in _activity_calls
        assert "complete_rebuild_job_activity" not in _activity_calls

    async def test_swap_failure_marks_job_failed_and_releases(self) -> None:
        @activity.defn(name="swap_index_activity")
        async def mock_swap_fail(input: SwapIndexInput) -> None:
            _activity_calls.append("swap_index_activity")
            raise ApplicationError("Failed to swap index", non_retryable=True)

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_swap_index"]
        activities.append(mock_swap_fail)

        with pytest.raises(WorkflowFailureError):
            await _run(activities, _manual_input("rebuild-swap-fail"))

        assert "fail_rebuild_job_activity" in _activity_calls
        assert "release_rebuild_claim_activity" in _activity_calls

    async def test_release_failure_does_not_mask_build_failure(self) -> None:
        @activity.defn(name="build_index_activity")
        async def mock_build_fail(input: BuildIndexInput) -> BuildIndexOutput:
            _activity_calls.append("build_index_activity")
            raise ApplicationError("Build failed: no embeddings", non_retryable=True)

        @activity.defn(name="release_rebuild_claim_activity")
        async def mock_release_fail(input: ReleaseRebuildClaimInput) -> None:
            _activity_calls.append("release_rebuild_claim_activity")
            raise ApplicationError("release exploded", non_retryable=True)

        activities = [
            a
            for a in ALL_MOCK_ACTIVITIES
            if a.__name__ not in ("mock_build_index", "mock_release_claim")
        ]
        activities.extend([mock_build_fail, mock_release_fail])

        with pytest.raises(WorkflowFailureError) as excinfo:
            await _run(activities, _manual_input("rebuild-fail"))

        assert "Build failed" in str(excinfo.value.cause.__cause__)
        assert "fail_rebuild_job_activity" in _activity_calls
        assert "release_rebuild_claim_activity" in _activity_calls


class TestRebuildWorkflowGarbageCollect:
    async def test_empty_removed_versions(self) -> None:
        @activity.defn(name="garbage_collect_indexes_activity")
        async def mock_gc_empty() -> GarbageCollectOutput:
            _activity_calls.append("garbage_collect_indexes_activity")
            return GarbageCollectOutput(removed_versions=[])

        activities = [a for a in ALL_MOCK_ACTIVITIES if a.__name__ != "mock_gc_indexes"]
        activities.append(mock_gc_empty)

        result = await _run(activities, _manual_input("rebuild-no-gc"))

        assert result.removed_versions == []
