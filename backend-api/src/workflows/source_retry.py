from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.ingestion.activities import finalize_source_run_activity
    from activities.ingestion.models import FinalizeSourceRunInput
    from shared.models import SourceRunStatus
    from workflows.ingest import IngestWorkflow
    from workflows.models import (
        IngestWorkflowInput,
        SourceRetryWorkflowInput,
        SourceRetryWorkflowOutput,
    )

RETRY_DB = RetryPolicy(
    maximum_attempts=5,
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=10),
)


@workflow.defn
class SourceRetryWorkflow:
    @workflow.run
    async def run(self, input: SourceRetryWorkflowInput) -> SourceRetryWorkflowOutput:
        await workflow.execute_child_workflow(
            IngestWorkflow.run,
            IngestWorkflowInput(job_id=input.job_id, dataset=input.dataset),
            id=f"ingest-workflow-{input.job_id}",
        )

        statuses: list[SourceRunStatus] = []
        for source_run_id in input.source_run_ids:
            final = await workflow.execute_activity(
                finalize_source_run_activity,
                FinalizeSourceRunInput(source_run_id=source_run_id),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )
            statuses.append(final.status)

        return SourceRetryWorkflowOutput(
            job_id=input.job_id,
            source_run_ids=input.source_run_ids,
            statuses=statuses,
        )
