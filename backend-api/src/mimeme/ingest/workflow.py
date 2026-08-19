from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from mimeme.ingest import rule
    from mimeme.ingest.model import BatchInput, Finish, Input, ItemRef, WorkflowInput
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
        if not workflow.patched("batched-siglip2-embeddings"):
            return await self._run_items(input)

        semaphore = asyncio.Semaphore(rule.FANOUT)

        async def one(refs: list[ItemRef]) -> None:
            async with semaphore:
                await workflow.execute_activity(
                    rule.BATCH_ACTIVITY,
                    BatchInput(
                        items=[
                            Input(
                                job_id=input.job_id,
                                item_id=ref.item_id,
                                source=ref.source,
                                dataset=input.dataset,
                            )
                            for ref in refs
                        ]
                    ),
                    start_to_close_timeout=timedelta(minutes=15),
                    heartbeat_timeout=timedelta(seconds=30),
                    retry_policy=_ITEM_RETRY,
                )

        batches = [
            input.items[offset : offset + rule.EMBED_BATCH_SIZE]
            for offset in range(0, len(input.items), rule.EMBED_BATCH_SIZE)
        ]
        await asyncio.gather(*(one(batch) for batch in batches), return_exceptions=True)

        return await workflow.execute_activity(
            rule.FINISH_ACTIVITY,
            Finish(job_id=input.job_id),
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=_FINISH_RETRY,
        )

    async def _run_items(self, input: WorkflowInput) -> IngestResult:
        """Replay-compatible path for workflows started before batching shipped."""
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
