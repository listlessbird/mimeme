from __future__ import annotations

from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from mimeme.ingest import rule
from mimeme.ingest.model import Finish, Input, ItemRef, RemoteUrl, Result, WorkflowInput
from mimeme.ingest.workflow import IngestWorkflow
from mimeme.job.model import IngestResult


def _items(n: int) -> list[ItemRef]:
    return [ItemRef(item_id=i, source=RemoteUrl(url=f"https://a/{i}.jpg")) for i in range(n)]


async def test_workflow_fans_out_then_finishes() -> None:
    seen: list[int] = []
    active = 0
    max_active = 0

    @activity.defn(name=rule.ITEM_ACTIVITY)
    async def item(inp: Input) -> Result:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        seen.append(inp.item_id)
        active -= 1
        return Result(item_id=inp.item_id, outcome="processed", image_id=inp.item_id)

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
            activities=[item, finish],
        ):
            result = await env.client.execute_workflow(
                IngestWorkflow.run,
                WorkflowInput(job_id="ingest-x", dataset=None, items=_items(10)),
                id=rule.workflow_id("ingest-x"),
                task_queue=rule.TASK_QUEUE,
            )

    assert sorted(seen) == list(range(10))
    assert result.processed == 10
    assert max_active <= rule.FANOUT


async def test_item_failure_is_tolerated_and_retried() -> None:
    attempts: dict[int, int] = {}

    @activity.defn(name=rule.ITEM_ACTIVITY)
    async def item(inp: Input) -> Result:
        attempts[inp.item_id] = attempts.get(inp.item_id, 0) + 1
        if inp.item_id == 1:
            raise RuntimeError("infra boom")
        return Result(item_id=inp.item_id, outcome="processed", image_id=inp.item_id)

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
            activities=[item, finish],
        ):
            result = await env.client.execute_workflow(
                IngestWorkflow.run,
                WorkflowInput(job_id="ingest-y", dataset=None, items=_items(3)),
                id=rule.workflow_id("ingest-y"),
                task_queue=rule.TASK_QUEUE,
            )

    # the failing item exhausts its retry policy (3 attempts) but the workflow
    # still reaches finish and completes.
    assert attempts[1] == 3
    assert result.failed == 1
