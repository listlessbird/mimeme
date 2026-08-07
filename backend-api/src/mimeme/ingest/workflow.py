from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from mimeme.ingest import rule
    from mimeme.ingest.model import Finish, Input, ItemRef, WorkflowInput
    from mimeme.job.model import IngestResult

_ITEM_RETRY = RetryPolicy(
    maximum_attempts=3,
    initial_interval=timedelta(seconds=2),
    maximum_interval=timedelta(seconds=30),
)
_FINISH_RETRY = RetryPolicy(
    maximum_attempts=5,
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=10),
)


@workflow.defn(name=rule.WORKFLOW)
class IngestWorkflow:
    @workflow.run
    async def run(self, input: WorkflowInput) -> IngestResult:
        semaphore = asyncio.Semaphore(rule.FANOUT)

        async def one(ref: ItemRef) -> None:
            async with semaphore:
                await workflow.execute_activity(
                    rule.ITEM_ACTIVITY,
                    Input(
                        job_id=input.job_id,
                        item_id=ref.item_id,
                        source=ref.source,
                        dataset=input.dataset,
                    ),
                    start_to_close_timeout=timedelta(minutes=15),
                    heartbeat_timeout=timedelta(seconds=30),
                    retry_policy=_ITEM_RETRY,
                )

        await asyncio.gather(*(one(ref) for ref in input.items), return_exceptions=True)

        return await workflow.execute_activity(
            rule.FINISH_ACTIVITY,
            Finish(job_id=input.job_id),
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=_FINISH_RETRY,
        )
