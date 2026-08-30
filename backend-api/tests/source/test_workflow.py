from __future__ import annotations

from datetime import timedelta

import pytest
from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from mimeme.db.schema import SourceRunStatus, SourceRunTrigger
    from mimeme.ingest import rule as ingest_rule
    from mimeme.ingest.model import ItemRef, RemoteUrl
    from mimeme.ingest.model import WorkflowInput as IngestWorkflowInput
    from mimeme.job.model import IngestResult
    from mimeme.source import rule
    from mimeme.source.model import (
        CleanupInput,
        DiscoverInput,
        DiscoverResult,
        FinishInput,
        FinishResult,
        RetryInput,
        SyncInput,
    )
    from mimeme.source.workflow import SourceRetryWorkflow, SourceSyncWorkflow

# Mutated by an *activity* (runs in the worker process, not the workflow
# sandbox), so the test can observe which child workflow ids actually ran.
_CHILD_IDS: list[str] = []


@activity.defn(name=rule.CHECKPOINT_CLEANUP_ACTIVITY)
async def cleanup_checkpoint(input: CleanupInput) -> None:
    del input


@activity.defn(name="test.record_child")
async def record_child(child_id: str) -> None:
    _CHILD_IDS.append(child_id)


@workflow.defn(name=ingest_rule.WORKFLOW)
class FakeIngestWorkflow:
    @workflow.run
    async def run(self, input: IngestWorkflowInput) -> IngestResult:
        await workflow.execute_activity(
            "test.record_child",
            workflow.info().workflow_id,
            start_to_close_timeout=timedelta(seconds=10),
        )
        return IngestResult(processed=len(input.items), failed=0, duplicates=0)


@workflow.defn(name=ingest_rule.WORKFLOW)
class FailingIngestWorkflow:
    @workflow.run
    async def run(self, input: IngestWorkflowInput) -> IngestResult:
        raise ApplicationError("ingest child boom")


def _refs(n: int) -> list[ItemRef]:
    return [ItemRef(item_id=i, source=RemoteUrl(url=f"https://a/{i}.jpg")) for i in range(n)]


def _cause_chain(exc: BaseException) -> str:
    parts: list[str] = []
    cur: BaseException | None = exc
    while cur is not None:
        parts.append(str(cur))
        cur = cur.__cause__
    return " | ".join(parts)


def _discover_activity(result: DiscoverResult, seen_triggers: list[SourceRunTrigger]):
    @activity.defn(name=rule.DISCOVER_ACTIVITY)
    async def discover(input: DiscoverInput) -> DiscoverResult:
        seen_triggers.append(input.trigger)
        return result

    return discover


def _finish_activity(seen: list[FinishInput]):
    @activity.defn(name=rule.FINISH_ACTIVITY)
    async def finish(input: FinishInput) -> FinishResult:
        seen.append(input)
        status = SourceRunStatus.FAILED if input.error else SourceRunStatus.COMPLETED
        return FinishResult(status=status, discovered=1, queued=1, duplicate=0, failed=0)

    return finish


async def _run(workflows, activities, run_call):
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        async with Worker(
            env.client,
            task_queue=rule.TASK_QUEUE,
            workflows=workflows,
            activities=[*activities, cleanup_checkpoint],
        ):
            return await run_call(env)


class TestSourceSync:
    async def test_manual_sync_runs_child_and_finishes(self) -> None:
        _CHILD_IDS.clear()
        triggers: list[SourceRunTrigger] = []
        finishes: list[FinishInput] = []
        discovered = DiscoverResult(
            source_run_id=7,
            ingest_job_id="ingest-abc",
            dataset="d",
            items=_refs(2),
            discovered=2,
            queued=2,
        )

        async def call(env):
            return await env.client.execute_workflow(
                SourceSyncWorkflow.run,
                SyncInput(source_id=1, trigger=SourceRunTrigger.MANUAL),
                id=rule.manual_workflow_id(1, "req"),
                task_queue=rule.TASK_QUEUE,
            )

        result = await _run(
            [SourceSyncWorkflow, FakeIngestWorkflow],
            [_discover_activity(discovered, triggers), _finish_activity(finishes), record_child],
            call,
        )
        assert result.status == SourceRunStatus.COMPLETED
        assert result.ingest_job_id == "ingest-abc" and result.source_run_id == 7
        assert triggers == [SourceRunTrigger.MANUAL]
        assert finishes == [FinishInput(source_run_id=7, error=None)]
        assert _CHILD_IDS == [ingest_rule.workflow_id("ingest-abc")]

    async def test_scheduled_trigger_flows_through(self) -> None:
        triggers: list[SourceRunTrigger] = []
        finishes: list[FinishInput] = []
        discovered = DiscoverResult(
            source_run_id=1, ingest_job_id=None, dataset=None, items=[], discovered=0, queued=0
        )

        async def call(env):
            return await env.client.execute_workflow(
                SourceSyncWorkflow.run,
                SyncInput(source_id=1, trigger=SourceRunTrigger.SCHEDULED),
                id=rule.scheduled_workflow_id(1),
                task_queue=rule.TASK_QUEUE,
            )

        result = await _run(
            [SourceSyncWorkflow, FakeIngestWorkflow],
            [_discover_activity(discovered, triggers), _finish_activity(finishes), record_child],
            call,
        )
        # no job -> no child; finish still runs.
        assert triggers == [SourceRunTrigger.SCHEDULED]
        assert result.status == SourceRunStatus.COMPLETED
        assert finishes[0].error is None

    async def test_child_failure_preserves_original_error(self) -> None:
        triggers: list[SourceRunTrigger] = []
        finishes: list[FinishInput] = []
        discovered = DiscoverResult(
            source_run_id=5,
            ingest_job_id="ingest-x",
            dataset=None,
            items=_refs(1),
            discovered=1,
            queued=1,
        )

        async def call(env):
            return await env.client.execute_workflow(
                SourceSyncWorkflow.run,
                SyncInput(source_id=1),
                id=rule.manual_workflow_id(1, "boom"),
                task_queue=rule.TASK_QUEUE,
            )

        with pytest.raises(WorkflowFailureError) as excinfo:
            await _run(
                [SourceSyncWorkflow, FailingIngestWorkflow],
                [_discover_activity(discovered, triggers), _finish_activity(finishes)],
                call,
            )
        # finish was still called with an error (bookkeeping did not swallow the
        # failure), and the workflow surfaced the original child failure.
        assert len(finishes) == 1 and finishes[0].error is not None
        assert "boom" in _cause_chain(excinfo.value)


class TestSourceRetry:
    async def test_retry_runs_child_then_finishes_each_run(self) -> None:
        _CHILD_IDS.clear()
        finishes: list[FinishInput] = []

        async def call(env):
            return await env.client.execute_workflow(
                SourceRetryWorkflow.run,
                RetryInput(
                    job_id="ingest-retry",
                    dataset="d",
                    source_run_ids=[3, 4],
                    items=_refs(2),
                ),
                id=rule.retry_workflow_id(3, "req"),
                task_queue=rule.TASK_QUEUE,
            )

        result = await _run(
            [SourceRetryWorkflow, FakeIngestWorkflow],
            [_finish_activity(finishes), record_child],
            call,
        )
        assert result.job_id == "ingest-retry"
        assert [f.source_run_id for f in finishes] == [3, 4]
        assert result.statuses == [SourceRunStatus.COMPLETED, SourceRunStatus.COMPLETED]
        assert _CHILD_IDS == [ingest_rule.workflow_id("ingest-retry")]
