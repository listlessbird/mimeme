from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.sources import (
        CompleteSourceRunInput,
        CreateSourceIngestInput,
        CreateSourceRunInput,
        CreateSourceRunOutput,
        FetchSourceItemsInput,
        FetchSourceItemsOutput,
        FilterSeenItemsInput,
        FilterSeenItemsOutput,
        PersistSourceItemsInput,
        PersistSourceItemsOutput,
    )
    from shared.config import settings
    from shared.models.orm import SourceRunStatus
    from workflows.models import (
        IngestWorkflowInput,
        IngestWorkflowOutput,
        SourceSyncWorkflowInput,
        SourceSyncWorkflowOutput,
    )


@workflow.defn
class SourceSyncWorkflow:
    @workflow.run
    async def run(self, input: SourceSyncWorkflowInput) -> SourceSyncWorkflowOutput:
        source_run_id: int | None = None
        discovered = 0
        seen = 0
        queued = 0
        duplicates = 0
        failed = 0
        skipped = 0
        error_message: str | None = None

        try:
            source_cfg = await workflow.execute_activity(
                "load_source_config_activity",
                input.source_id,
                task_queue=settings.temporal_task_queue,
                start_to_close_timeout=timedelta(minutes=1),
            )

            run_out = CreateSourceRunOutput.model_validate(
                await workflow.execute_activity(
                    "create_source_run_activity",
                    CreateSourceRunInput(
                        source_id=input.source_id, trigger_mode=input.trigger_mode
                    ),
                    task_queue=settings.temporal_task_queue,
                    start_to_close_timeout=timedelta(minutes=1),
                )
            )

            source_run_id = run_out.source_run_id
            fetch_result = FetchSourceItemsOutput.model_validate(
                await workflow.execute_activity(
                    "fetch_source_items_activity",
                    FetchSourceItemsInput(
                        source_id=input.source_id,
                        adapter_key=source_cfg.adapter_key,
                        adapter_config=source_cfg.adapter_config,
                        secret_refs=source_cfg.secret_refs,
                        max_items_per_run=source_cfg.max_items_per_run,
                    ),
                    task_queue=settings.temporal_task_queue,
                    start_to_close_timeout=timedelta(minutes=10),
                    retry_policy=RetryPolicy(
                        maximum_attempts=2, initial_interval=timedelta(seconds=5)
                    ),
                )
            )

        except Exception as e:
            error_message = str(e)
