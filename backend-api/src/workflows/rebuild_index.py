from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import json
    from datetime import UTC, datetime

    from activities.indexing import (
        BuildIndexInput,
        BuildIndexOutput,
        GarbageCollectOutput,
        SwapIndexInput,
    )
    from shared.config import settings
    from shared.db import session_scope
    from shared.models import Job, JobStatus
    from workflows.models import RebuildIndexWorkflowInput, RebuildIndexWorkflowOutput


@workflow.defn
class RebuildIndexWorkflow:
    @workflow.run
    async def run(self, input: RebuildIndexWorkflowInput) -> RebuildIndexWorkflowOutput:
        with session_scope() as session:
            job = session.query(Job).filter_by(id=input.job_id).first()
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now(UTC)

        model_name = input.model_name or settings.embed_model

        self._update_progress(input.job_id, 10, "Building index...")

        build_result: BuildIndexOutput = await workflow.execute_activity(
            "build_index_activity",
            BuildIndexInput(
                model_name=model_name,
                index_type=settings.index_type,
                force=input.force,
            ),
            task_queue=settings.temporal_task_queue_cpu,
            start_to_close_timeout=timedelta(hours=2),
            heartbeat_timeout=timedelta(minutes=5),
        )

        self._update_progress(input.job_id, 70, "Swapping to new index...")

        await workflow.execute_activity(
            "swap_index_activity",
            SwapIndexInput(version=build_result.version),
            task_queue=settings.temporal_task_queue_cpu,
            start_to_close_timeout=timedelta(minutes=5),
        )

        self._update_progress(input.job_id, 90, "Cleaning up old indexes...")

        gc_result: GarbageCollectOutput = await workflow.execute_activity(
            "garbage_collect_indexes_activity",
            task_queue=settings.temporal_task_queue_cpu,
            start_to_close_timeout=timedelta(minutes=10),
        )

        self._complete_job(input.job_id, build_result, gc_result.removed_versions)

        return RebuildIndexWorkflowOutput(
            job_id=input.job_id,
            version=build_result.version,
            num_vectors=build_result.num_vectors,
            dimension=build_result.dimension,
            removed_versions=gc_result.removed_versions,
        )

    def _update_progress(self, job_id: str, progress: float, message: str) -> None:
        with session_scope() as session:
            job = session.query(Job).filter_by(id=job_id).first()
            if job:
                job.progress = progress
                job.message = message

    def _complete_job(
        self, job_id: str, build_result: BuildIndexOutput, removed: list[str]
    ) -> None:
        with session_scope() as session:
            job = session.query(Job).filter_by(id=job_id).first()
            if job:
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.completed_at = datetime.now(UTC)
                job.result = json.dumps(
                    {
                        "version": build_result.version,
                        "num_vectors": build_result.num_vectors,
                        "dimension": build_result.dimension,
                        "removed_versions": removed,
                    }
                )
