from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.indexing.activities import (
        build_index_activity,
        garbage_collect_indexes_activity,
        swap_index_activity,
    )
    from activities.indexing.models import (
        BuildIndexInput,
        BuildIndexOutput,
        GarbageCollectOutput,
        SwapIndexInput,
    )
    from activities.workflow_state.activities import (
        complete_rebuild_job_activity,
        fail_rebuild_job_activity,
        prepare_rebuild_activity,
        reconcile_generation_activity,
        release_rebuild_claim_activity,
        start_rebuild_job_activity,
        update_job_progress_activity,
    )
    from activities.workflow_state.models import (
        CompleteRebuildJobInput,
        FailRebuildJobInput,
        PrepareRebuildInput,
        PrepareRebuildOutput,
        ReconcileGenerationInput,
        ReleaseRebuildClaimInput,
        StartRebuildJobInput,
        UpdateJobProgressInput,
    )
    from shared.models import RebuildTrigger
    from workflows.models import RebuildIndexWorkflowInput, RebuildIndexWorkflowOutput

RETRY_DB = RetryPolicy(
    maximum_attempts=5,
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=10),
)

RETRY_INDEX_BUILD = RetryPolicy(maximum_attempts=1)

BUSY_RETRY_INTERVAL = timedelta(seconds=30)


@workflow.defn
class RebuildIndexWorkflow:
    @workflow.run
    async def run(self, input: RebuildIndexWorkflowInput) -> RebuildIndexWorkflowOutput:
        prepare_input = PrepareRebuildInput(
            job_id=input.job_id,
            workflow_id=workflow.info().workflow_id,
            force=input.force,
            trigger=input.trigger,
        )

        prep: PrepareRebuildOutput = await workflow.execute_activity(
            prepare_rebuild_activity,
            prepare_input,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=RETRY_DB,
        )

        while prep.decision == "busy" and input.trigger == RebuildTrigger.MANUAL:
            await workflow.sleep(BUSY_RETRY_INTERVAL)
            prep = await workflow.execute_activity(
                prepare_rebuild_activity,
                prepare_input,
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )

        if prep.decision == "busy":
            return RebuildIndexWorkflowOutput(job_id=None, outcome="busy")

        if prep.decision == "clean":
            return RebuildIndexWorkflowOutput(job_id=prep.job_id, outcome="skipped")

        assert prep.job_id is not None and prep.target_generation is not None
        job_id = prep.job_id
        target_generation = prep.target_generation

        last_step = "start_job"
        outcome = "built"
        claimed = True
        swap_completed = False
        build_result: BuildIndexOutput | None = None
        gc_result: GarbageCollectOutput | None = None
        error_message: str | None = None

        try:
            await workflow.execute_activity(
                start_rebuild_job_activity,
                StartRebuildJobInput(job_id=job_id),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )

            await workflow.execute_activity(
                update_job_progress_activity,
                UpdateJobProgressInput(job_id=job_id, progress=10, message="Building index..."),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )

            last_step = "build_index"
            build_result = await workflow.execute_activity(
                build_index_activity,
                BuildIndexInput(
                    model_name=input.model_name,
                    index_type=input.index_type,
                    force=input.force,
                    target_generation=target_generation,
                ),
                start_to_close_timeout=timedelta(hours=2),
                retry_policy=RETRY_INDEX_BUILD,
            )

            if build_result.outcome == "empty_reconcile":
                last_step = "reconcile_generation"
                outcome = "empty_reconcile"
                await workflow.execute_activity(
                    reconcile_generation_activity,
                    ReconcileGenerationInput(job_id=job_id, target_generation=target_generation),
                    start_to_close_timeout=timedelta(minutes=1),
                    retry_policy=RETRY_DB,
                )
                swap_completed = True
                await workflow.execute_activity(
                    complete_rebuild_job_activity,
                    CompleteRebuildJobInput(
                        job_id=job_id,
                        version="",
                        num_vectors=0,
                        dimension=0,
                        removed_versions=[],
                        text_num_vectors=None,
                    ),
                    start_to_close_timeout=timedelta(minutes=1),
                    retry_policy=RETRY_DB,
                )
                await self._release(job_id)
                claimed = False
                return RebuildIndexWorkflowOutput(
                    job_id=job_id, outcome="empty_reconcile", num_vectors=0
                )

            await workflow.execute_activity(
                update_job_progress_activity,
                UpdateJobProgressInput(
                    job_id=job_id, progress=70, message="Swapping to new index..."
                ),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )

            last_step = "swap_index"
            assert build_result.version is not None
            await workflow.execute_activity(
                swap_index_activity,
                SwapIndexInput(
                    version=build_result.version,
                    job_id=job_id,
                    target_generation=target_generation,
                ),
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RETRY_DB,
            )
            swap_completed = True

            await workflow.execute_activity(
                update_job_progress_activity,
                UpdateJobProgressInput(
                    job_id=job_id, progress=90, message="Cleaning up old indexes..."
                ),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )

            last_step = "gc_indexes"
            gc_result = await workflow.execute_activity(
                garbage_collect_indexes_activity,
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=RETRY_DB,
            )

            last_step = "complete_job"
            await workflow.execute_activity(
                complete_rebuild_job_activity,
                CompleteRebuildJobInput(
                    job_id=job_id,
                    version=build_result.version,
                    num_vectors=build_result.num_vectors,
                    dimension=build_result.dimension or 0,
                    removed_versions=gc_result.removed_versions,
                    text_num_vectors=build_result.text_num_vectors,
                ),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_DB,
            )

            await self._release(job_id)
            claimed = False

            return RebuildIndexWorkflowOutput(
                job_id=job_id,
                outcome="built",
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
                    fail_rebuild_job_activity,
                    FailRebuildJobInput(
                        job_id=job_id,
                        error=f"Failed at step '{last_step}': {error_message}",
                    ),
                    start_to_close_timeout=timedelta(minutes=1),
                    retry_policy=RETRY_DB,
                )
            except Exception:
                workflow.logger.error(
                    "fail_rebuild_job_activity_error",
                    extra={"job_id": job_id, "original_error": error_message},
                )
            if claimed:
                try:
                    await self._release(job_id)
                except Exception:
                    workflow.logger.error(
                        "release_rebuild_claim_error",
                        extra={"job_id": job_id, "original_error": error_message},
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
                    "job_id": job_id,
                    "trigger": input.trigger.value,
                    "force": input.force,
                    "model_name": input.model_name,
                    "index_type": input.index_type,
                    "target_generation": target_generation,
                    "last_step": last_step,
                    "swap_completed": swap_completed,
                    "outcome": "error" if error_message else outcome,
                    "error": error_message,
                    "version": build_result.version if build_result else None,
                    "num_vectors": build_result.num_vectors if build_result else None,
                    "dimension": build_result.dimension if build_result else None,
                    "text_num_vectors": build_result.text_num_vectors if build_result else None,
                    "removed_versions": len(gc_result.removed_versions) if gc_result else None,
                },
            )

    async def _release(self, job_id: str) -> None:
        await workflow.execute_activity(
            release_rebuild_claim_activity,
            ReleaseRebuildClaimInput(job_id=job_id),
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=RETRY_DB,
        )
