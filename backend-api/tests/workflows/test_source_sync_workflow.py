"""Integration tests for SourceSyncWorkflow using Temporal's test environment.

The workflow orchestrates one Source run: open the run, plan + fetch (the shuffle
runs in-workflow via workflow.random()), discover/queue, await the ingest as a
child workflow, then finalize — marking the run FAILED on a hard error.

Activities and the child IngestWorkflow are mocked by name; the real
SourceSyncWorkflow (including build_requests planning) runs.
"""

from __future__ import annotations

import uuid

import pytest
from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.ingestion.models import (
    DiscoverAndQueueInput,
    DiscoverAndQueueOutput,
    FailSourceRunInput,
    FetchSourceInput,
    FetchSourceOutput,
    FinalizeSourceRunInput,
    FinalizeSourceRunOutput,
    StartSourceRunInput,
    StartSourceRunOutput,
)
from shared.models import SourceRunStatus
from workflows.models import (
    IngestWorkflowInput,
    IngestWorkflowOutput,
    SourceSyncWorkflowInput,
    SourceSyncWorkflowOutput,
)
from workflows.source_sync import SourceSyncWorkflow

# ---------------------------------------------------------------------------
# Mock activities + child workflow
# ---------------------------------------------------------------------------

_calls: list[str] = []


class _State:
    ingest_job_id: str | None = "ingest-abc"
    discover_raises: bool = False


_state = _State()


@activity.defn(name="start_source_run_activity")
async def mock_start(input: StartSourceRunInput) -> StartSourceRunOutput:
    _calls.append("start_source_run_activity")
    return StartSourceRunOutput(
        source_run_id=1,
        adapter_key="meme_api",
        adapter_config={"subreddits": ["memes", "aww"]},
        max_items_per_run=20,
        dataset="ds",
    )


@activity.defn(name="fetch_source_activity")
async def mock_fetch(input: FetchSourceInput) -> FetchSourceOutput:
    _calls.append("fetch_source_activity")
    return FetchSourceOutput(success=True, status_code=200, raw={"memes": []})


@activity.defn(name="discover_and_queue_activity")
async def mock_discover(input: DiscoverAndQueueInput) -> DiscoverAndQueueOutput:
    _calls.append("discover_and_queue_activity")
    if _state.discover_raises:
        raise ApplicationError("discover boom", non_retryable=True)
    return DiscoverAndQueueOutput(discovered=2, queued=2, ingest_job_id=_state.ingest_job_id)


@activity.defn(name="finalize_source_run_activity")
async def mock_finalize(input: FinalizeSourceRunInput) -> FinalizeSourceRunOutput:
    _calls.append("finalize_source_run_activity")
    return FinalizeSourceRunOutput(
        status=SourceRunStatus.COMPLETED, discovered=2, queued=2, duplicate=0, failed=0
    )


@activity.defn(name="fail_source_run_activity")
async def mock_fail(input: FailSourceRunInput) -> None:
    _calls.append("fail_source_run_activity")


# Unsandboxed: this mock lives in the test module, so sandboxing it would make
# Temporal re-import the test module (and its heavy activity import chain) inside
# the workflow sandbox. The real SourceSyncWorkflow under test stays sandboxed.
@workflow.defn(name="IngestWorkflow", sandboxed=False)
class MockIngestWorkflow:
    @workflow.run
    async def run(self, input: IngestWorkflowInput) -> IngestWorkflowOutput:
        return IngestWorkflowOutput(
            job_id=input.job_id, total=1, processed=1, failed=0, duplicates=0
        )


ALL_MOCK_ACTIVITIES = [mock_start, mock_fetch, mock_discover, mock_finalize, mock_fail]


@pytest.fixture(autouse=True)
def _reset() -> None:
    _calls.clear()
    _state.ingest_job_id = "ingest-abc"
    _state.discover_raises = False


async def _run() -> SourceSyncWorkflowOutput:
    task_queue = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[SourceSyncWorkflow, MockIngestWorkflow],
            activities=ALL_MOCK_ACTIVITIES,
        ):
            return await env.client.execute_workflow(
                SourceSyncWorkflow.run,
                SourceSyncWorkflowInput(source_id=42),
                id=str(uuid.uuid4()),
                task_queue=task_queue,
            )


class TestSourceSyncWorkflow:
    async def test_happy_path_runs_child_and_finalizes(self) -> None:
        result = await _run()

        assert result.source_run_id == 1
        assert result.status == SourceRunStatus.COMPLETED
        assert result.ingest_job_id == "ingest-abc"
        assert result.queued == 2
        # The whole pipeline ran in order; the child Ingest dispatched (else the
        # workflow would have errored on an unregistered child).
        assert _calls == [
            "start_source_run_activity",
            "fetch_source_activity",
            "discover_and_queue_activity",
            "finalize_source_run_activity",
        ]

    async def test_no_new_items_skips_child_and_still_finalizes(self) -> None:
        _state.ingest_job_id = None

        result = await _run()

        assert result.ingest_job_id is None
        assert result.status == SourceRunStatus.COMPLETED
        assert "finalize_source_run_activity" in _calls

    async def test_hard_failure_marks_run_failed_and_reraises(self) -> None:
        _state.discover_raises = True

        with pytest.raises(WorkflowFailureError):
            await _run()

        assert "fail_source_run_activity" in _calls
        assert "finalize_source_run_activity" not in _calls
