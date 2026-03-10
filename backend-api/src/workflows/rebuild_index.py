from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities.indexing import (
        BuildIndexInput,
        BuildIndexOutput,
        GarbageCollectOutput,
        SwapIndexInput,
    )
    from activities.workflow_state import (
        CompleteRebuildJobInput,
        FailRebuildJobInput,
        StartRebuildJobInput,
        UpdateJobProgressInput,
    )
    from shared.config import settings
    from workflows.models import RebuildIndexWorkflowInput, RebuildIndexWorkflowOutput


@workflow.defn
class RebuildIndexWorkflow:
    @workflow.run
    async def run(self, input: RebuildIndexWorkflowInput) -> RebuildIndexWorkflowOutput:
        model_name = input.model_name or settings.embed_model
        last_step = "start"
        build_result: BuildIndexOutput | None = None
        gc_result: GarbageCollectOutput | None = None
        error_message: str | None = None

        try:
            last_step = "start_job"
            workflow.logger.info(
                "workflow_step",
                extra={
                    "workflow_name": "RebuildIndexWorkflow",
                    "job_id": input.job_id,
                    "step": "start_job",
                },
            )
            await workflow.execute_activity(
                "start_rebuild_job_activity",
                StartRebuildJobInput(job_id=input.job_id),
                start_to_close_timeout=timedelta(minutes=1),
            )

            last_step = "progress_10"
            workflow.logger.info(
                "workflow_step",
                extra={
                    "workflow_name": "RebuildIndexWorkflow",
                    "job_id": input.job_id,
                    "step": "update_progress_10",
                },
            )
            await workflow.execute_activity(
                "update_job_progress_activity",
                UpdateJobProgressInput(
                    job_id=input.job_id,
                    progress=10,
                    message="Building index...",
                ),
                start_to_close_timeout=timedelta(minutes=1),
            )

            last_step = "build_index"
            workflow.logger.info(
                "workflow_step",
                extra={
                    "workflow_name": "RebuildIndexWorkflow",
                    "job_id": input.job_id,
                    "step": "build_index",
                },
            )
            build_result = BuildIndexOutput.model_validate(
                await workflow.execute_activity(
                    "build_index_activity",
                    BuildIndexInput(
                        model_name=model_name,
                        index_type=settings.index_type,
                        force=input.force,
                    ),
                    start_to_close_timeout=timedelta(hours=2),
                    heartbeat_timeout=timedelta(minutes=5),
                )
            )

            last_step = "progress_70"
            workflow.logger.info(
                "workflow_step",
                extra={
                    "workflow_name": "RebuildIndexWorkflow",
                    "job_id": input.job_id,
                    "step": "update_progress_70",
                },
            )
            await workflow.execute_activity(
                "update_job_progress_activity",
                UpdateJobProgressInput(
                    job_id=input.job_id,
                    progress=70,
                    message="Swapping to new index...",
                ),
                start_to_close_timeout=timedelta(minutes=1),
            )

            last_step = "swap_index"
            workflow.logger.info(
                "workflow_step",
                extra={
                    "workflow_name": "RebuildIndexWorkflow",
                    "job_id": input.job_id,
                    "version": build_result.version,
                    "step": "swap_index",
                },
            )
            await workflow.execute_activity(
                "swap_index_activity",
                SwapIndexInput(version=build_result.version),
                start_to_close_timeout=timedelta(minutes=5),
            )

            last_step = "progress_90"
            workflow.logger.info(
                "workflow_step",
                extra={
                    "workflow_name": "RebuildIndexWorkflow",
                    "job_id": input.job_id,
                    "step": "update_progress_90",
                },
            )
            await workflow.execute_activity(
                "update_job_progress_activity",
                UpdateJobProgressInput(
                    job_id=input.job_id,
                    progress=90,
                    message="Cleaning up old indexes...",
                ),
                start_to_close_timeout=timedelta(minutes=1),
            )

            last_step = "gc_indexes"
            workflow.logger.info(
                "workflow_step",
                extra={
                    "workflow_name": "RebuildIndexWorkflow",
                    "job_id": input.job_id,
                    "step": "garbage_collect_indexes",
                },
            )
            gc_result = GarbageCollectOutput.model_validate(
                await workflow.execute_activity(
                    "garbage_collect_indexes_activity",
                    start_to_close_timeout=timedelta(minutes=10),
                )
            )

            last_step = "complete_job"
            workflow.logger.info(
                "workflow_step",
                extra={
                    "workflow_name": "RebuildIndexWorkflow",
                    "job_id": input.job_id,
                    "step": "complete_job",
                },
            )
            await workflow.execute_activity(
                "complete_rebuild_job_activity",
                CompleteRebuildJobInput(
                    job_id=input.job_id,
                    version=build_result.version,
                    num_vectors=build_result.num_vectors,
                    dimension=build_result.dimension,
                    removed_versions=gc_result.removed_versions,
                    text_num_vectors=build_result.text_num_vectors,
                ),
                start_to_close_timeout=timedelta(minutes=1),
            )

            return RebuildIndexWorkflowOutput(
                job_id=input.job_id,
                version=build_result.version,
                num_vectors=build_result.num_vectors,
                dimension=build_result.dimension,
                removed_versions=gc_result.removed_versions,
                text_num_vectors=build_result.text_num_vectors,
            )
        except Exception as exc:
            error_message = str(exc)
            try:
                await workflow.execute_activity(
                    "fail_rebuild_job_activity",
                    FailRebuildJobInput(
                        job_id=input.job_id,
                        error=f"Failed at step '{last_step}': {error_message}",
                    ),
                    start_to_close_timeout=timedelta(minutes=1),
                )
            except Exception:
                workflow.logger.error(
                    "fail_rebuild_job_activity_error",
                    extra={"job_id": input.job_id, "original_error": error_message},
                )
            raise
        finally:
            workflow.logger.info(
                "workflow_wide_event",
                extra={
                    "event_type": "workflow_wide_event",
                    "workflow_name": "RebuildIndexWorkflow",
                    "workflow_id": workflow.info().workflow_id,
                    "run_id": workflow.info().run_id,
                    "job_id": input.job_id,
                    "model_name": model_name,
                    "force": input.force,
                    "last_step": last_step,
                    "outcome": "error" if error_message else "success",
                    "error": error_message,
                    "version": build_result.version if build_result else None,
                    "num_vectors": build_result.num_vectors if build_result else None,
                    "dimension": build_result.dimension if build_result else None,
                    "text_num_vectors": build_result.text_num_vectors if build_result else None,
                    "removed_versions": len(gc_result.removed_versions) if gc_result else None,
                },
            )
