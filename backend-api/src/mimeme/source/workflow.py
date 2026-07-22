from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from mimeme.ingest import rule as ingest_rule
    from mimeme.ingest.model import WorkflowInput as IngestWorkflowInput
    from mimeme.ingest.workflow import IngestWorkflow
    from mimeme.source import rule
    from mimeme.source.model import (
        DiscoverInput,
        DiscoverResult,
        FinishInput,
        FinishResult,
        RetryInput,
        RetryResult,
        SyncInput,
        SyncResult,
    )

_RETRY_DISCOVER = RetryPolicy(
    maximum_attempts=3,
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=30),
    non_retryable_error_types=["SourceNotFound", "UnknownAdapterKey"],
)
_RETRY_FINISH = RetryPolicy(
    maximum_attempts=5,
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=10),
)


@workflow.defn(name=rule.SYNC_WORKFLOW)
class SourceSyncWorkflow:
    @workflow.run
    async def run(self, input: SyncInput) -> SyncResult:
        discovered: DiscoverResult = await workflow.execute_activity(
            rule.DISCOVER_ACTIVITY,
            DiscoverInput(source_id=input.source_id, trigger=input.trigger),
            result_type=DiscoverResult,
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(seconds=rule.HEARTBEAT_TIMEOUT_S),
            retry_policy=_RETRY_DISCOVER,
        )

        try:
            if discovered.ingest_job_id is not None:
                await workflow.execute_child_workflow(
                    IngestWorkflow.run,
                    IngestWorkflowInput(
                        job_id=discovered.ingest_job_id,
                        dataset=discovered.dataset,
                        items=discovered.items,
                    ),
                    id=ingest_rule.workflow_id(discovered.ingest_job_id),
                )

            final = await workflow.execute_activity(
                rule.FINISH_ACTIVITY,
                FinishInput(source_run_id=discovered.source_run_id),
                result_type=FinishResult,
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=_RETRY_FINISH,
            )
        except Exception as exc:
            await workflow.execute_activity(
                rule.FINISH_ACTIVITY,
                FinishInput(source_run_id=discovered.source_run_id, error=str(exc)),
                result_type=FinishResult,
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=_RETRY_FINISH,
            )
            raise

        return SyncResult(
            source_run_id=discovered.source_run_id,
            status=final.status,
            discovered=final.discovered,
            queued=final.queued,
            duplicate=final.duplicate,
            failed=final.failed,
            ingest_job_id=discovered.ingest_job_id,
        )


@workflow.defn(name=rule.RETRY_WORKFLOW)
class SourceRetryWorkflow:
    @workflow.run
    async def run(self, input: RetryInput) -> RetryResult:
        await workflow.execute_child_workflow(
            IngestWorkflow.run,
            IngestWorkflowInput(job_id=input.job_id, dataset=input.dataset, items=input.items),
            id=ingest_rule.workflow_id(input.job_id),
        )

        statuses = []
        for source_run_id in input.source_run_ids:
            final = await workflow.execute_activity(
                rule.FINISH_ACTIVITY,
                FinishInput(source_run_id=source_run_id),
                result_type=FinishResult,
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=_RETRY_FINISH,
            )
            statuses.append(final.status)

        return RetryResult(
            job_id=input.job_id,
            source_run_ids=input.source_run_ids,
            statuses=statuses,
        )
