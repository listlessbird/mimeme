from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities.embedding import EncodeQueryInput, EncodeQueryOutput
    from shared.config import settings


@workflow.defn
class EncodeQueryWorkflow:
    @workflow.run
    async def run(self, query: str) -> EncodeQueryOutput:
        result: EncodeQueryOutput = await workflow.execute_activity(
            "encode_query_activity",
            EncodeQueryInput(query=query),
            task_queue=settings.temporal_task_queue_gpu,
            start_to_close_timeout=timedelta(seconds=30),
        )
        return result
