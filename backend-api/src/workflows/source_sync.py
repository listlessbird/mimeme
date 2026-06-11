from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.ingestion.activities import (
        discover_and_queue_activity,
        fail_source_run_activity,
        fetch_source_activity,
        finalize_source_run_activity,
        start_source_run_activity,
    )
    from activities.ingestion.models import (
        DiscoverAndQueueInput,
        FailSourceRunInput,
        FetchSourceInput,
        FinalizeSourceRunInput,
        StartSourceRunInput,
        StartSourceRunOutput,
    )
    from domain.adapters.registry import get_adapter
    from workflows.ingest import IngestWorkflow
    from workflows.models import (
        IngestWorkflowInput,
        SourceSyncWorkflowInput,
        SourceSyncWorkflowOutput,
    )

RETRY_NETWORK = RetryPolicy(
    maximum_attempts=3,
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=30),
)

RETRY_DB = RetryPolicy(
    maximum_attempts=5,
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=10),
)


@workflow.defn
class SourceSyncWorkflow:
    @workflow.run
    async def run(self, input: SourceSyncWorkflowInput) -> SourceSyncWorkflowOutput:
        # Open the run row first; if this fails there is nothing to finalize.
        start: StartSourceRunOutput = await workflow.execute_activity(
            start_source_run_activity,
            StartSourceRunInput(source_id=input.source_id, trigger=input.trigger),
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=RETRY_DB,
        )

        try:
            adapter = get_adapter(start.adapter_key)
            requests = adapter.build_requests(
                {**start.adapter_config, "max_items_per_run": start.max_items_per_run},
                rng=workflow.random(),
            )

            raw_responses: list[dict[str, Any]] = []
            for request in requests:
                fetched = await workflow.execute_activity(
                    fetch_source_activity,
                    FetchSourceInput(request=request, source_run_id=start.source_run_id),
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=RETRY_NETWORK,
                )
                if fetched.success and fetched.raw is not None:
                    raw_responses.append(fetched.raw)

            discovered = await workflow.execute_activity(
                discover_and_queue_activity,
                DiscoverAndQueueInput(
                    source_id=input.source_id,
                    source_run_id=start.source_run_id,
                    adapter_key=start.adapter_key,
                    dataset=start.dataset,
                    raw_responses=raw_responses,
                ),
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RETRY_DB,
            )

            # Reuse the existing ingest pipeline as an awaited child workflow.
            if discovered.ingest_job_id is not None:
                await workflow.execute_child_workflow(
                    IngestWorkflow.run,
                    IngestWorkflowInput(job_id=discovered.ingest_job_id, dataset=start.dataset),
                    id=f"ingest-workflow-{discovered.ingest_job_id}",
                )

            final = await workflow.execute_activity(
                finalize_source_run_activity,
                FinalizeSourceRunInput(source_run_id=start.source_run_id),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )

            return SourceSyncWorkflowOutput(
                source_run_id=start.source_run_id,
                status=final.status,
                discovered=final.discovered,
                queued=final.queued,
                duplicate=final.duplicate,
                failed=final.failed,
                ingest_job_id=discovered.ingest_job_id,
            )
        except Exception as exc:
            await workflow.execute_activity(
                fail_source_run_activity,
                FailSourceRunInput(source_run_id=start.source_run_id, error=str(exc)),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )
            raise
