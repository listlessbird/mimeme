from __future__ import annotations

from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from mimeme.ingest import rule
from mimeme.ingest.model import BatchInput, Finish, ItemRef, RemoteUrl, Result, WorkflowInput
from mimeme.ingest.workflow import IngestWorkflow
from mimeme.job.model import IngestResult


def _items(n: int) -> list[ItemRef]:
    return [ItemRef(item_id=i, source=RemoteUrl(url=f"https://a/{i}.jpg")) for i in range(n)]


async def test_workflow_fans_out_then_finishes() -> None:
    seen: list[int] = []
    active = 0
    max_active = 0

    batch_sizes: list[int] = []

    @activity.defn(name=rule.BATCH_ACTIVITY)
    async def batch(inp: BatchInput) -> list[Result]:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        batch_sizes.append(len(inp.items))
        seen.extend(item.item_id for item in inp.items)
        active -= 1
        return [
            Result(item_id=item.item_id, outcome="processed", image_id=item.item_id)
            for item in inp.items
        ]

    @activity.defn(name=rule.FINISH_ACTIVITY)
    async def finish(inp: Finish) -> IngestResult:
        return IngestResult(processed=len(seen), failed=0, duplicates=0)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        async with Worker(
            env.client,
            task_queue=rule.TASK_QUEUE,
            workflows=[IngestWorkflow],
            activities=[batch, finish],
        ):
            result = await env.client.execute_workflow(
                IngestWorkflow.run,
                WorkflowInput(job_id="ingest-x", dataset=None, items=_items(17)),
                id=rule.workflow_id("ingest-x"),
                task_queue=rule.TASK_QUEUE,
            )

    assert sorted(seen) == list(range(17))
    assert sorted(batch_sizes) == [1, 16]
    assert result.processed == 17
    assert max_active <= rule.FANOUT


async def test_transient_failure_retries_beyond_the_old_attempt_limit() -> None:
    attempts: dict[int, int] = {}

    @activity.defn(name=rule.BATCH_ACTIVITY)
    async def batch(inp: BatchInput) -> list[Result]:
        for item in inp.items:
            attempts[item.item_id] = attempts.get(item.item_id, 0) + 1
        if any(item.item_id == 1 for item in inp.items) and attempts[1] < 4:
            raise RuntimeError("infra boom")
        return [
            Result(item_id=item.item_id, outcome="processed", image_id=item.item_id)
            for item in inp.items
        ]

    @activity.defn(name=rule.FINISH_ACTIVITY)
    async def finish(inp: Finish) -> IngestResult:
        return IngestResult(processed=2, failed=1, duplicates=0)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        async with Worker(
            env.client,
            task_queue=rule.TASK_QUEUE,
            workflows=[IngestWorkflow],
            activities=[batch, finish],
        ):
            result = await env.client.execute_workflow(
                IngestWorkflow.run,
                WorkflowInput(job_id="ingest-y", dataset=None, items=_items(3)),
                id=rule.workflow_id("ingest-y"),
                task_queue=rule.TASK_QUEUE,
            )

    assert attempts[1] == 4
    assert result.failed == 1
