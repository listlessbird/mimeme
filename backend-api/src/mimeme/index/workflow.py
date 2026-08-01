from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from mimeme.index import rule
    from mimeme.index.model import (
        Activated,
        ActivateInput,
        Prepared,
        PrepareInput,
        Result,
        Trigger,
        WorkflowInput,
        WorkflowResult,
    )

_DB_RETRY = RetryPolicy(
    maximum_attempts=rule.PREPARE_MAX_ATTEMPTS,
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=10),
)
_BUILD_RETRY = RetryPolicy(
    maximum_attempts=rule.BUILD_MAX_ATTEMPTS,
    maximum_interval=timedelta(minutes=1),
)
_ACTIVATE_RETRY = RetryPolicy(
    maximum_attempts=rule.ACTIVATE_MAX_ATTEMPTS,
    maximum_interval=timedelta(seconds=10),
)


@workflow.defn(name=rule.WORKFLOW)
class RebuildWorkflow:
    @workflow.run
    async def run(self, input: WorkflowInput) -> WorkflowResult:
        job_id = input.job_id
        if input.trigger is Trigger.SCHEDULED and job_id is None:
            job_id = f"rebuild-{workflow.info().run_id.replace('-', '')[:12]}"
        prepared: Prepared = await workflow.execute_activity(
            rule.PREPARE_ACTIVITY,
            PrepareInput(
                job_id=job_id,
                workflow_id=workflow.info().workflow_id,
                force=input.force,
                trigger=input.trigger,
                model=input.model,
                index_type=input.index_type,
            ),
            result_type=Prepared,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=_DB_RETRY,
        )
        if prepared.decision == "busy":
            if input.trigger is Trigger.SCHEDULED:
                return WorkflowResult(job_id=None, outcome="busy")
            if input.busy_attempt + 1 >= rule.BUSY_ATTEMPTS_PER_RUN:
                await workflow.sleep(timedelta(seconds=rule.BUSY_WAIT_S))
                workflow.continue_as_new(input.model_copy(update={"busy_attempt": 0}))
            await workflow.sleep(timedelta(seconds=rule.BUSY_WAIT_S))
            return await self.run(input.model_copy(update={"busy_attempt": input.busy_attempt + 1}))
        if prepared.decision == "clean":
            return WorkflowResult(job_id=prepared.job_id, outcome="clean")

        assert prepared.build is not None and prepared.job_id is not None
        result: Result = await workflow.execute_activity(
            rule.BUILD_ACTIVITY,
            prepared.build,
            result_type=Result,
            start_to_close_timeout=timedelta(hours=2),
            heartbeat_timeout=timedelta(seconds=rule.HEARTBEAT_TIMEOUT_S),
            retry_policy=_BUILD_RETRY,
        )
        activated: Activated = await workflow.execute_activity(
            rule.ACTIVATE_ACTIVITY,
            ActivateInput(
                job_id=prepared.job_id,
                target_generation=prepared.build.target_generation,
                result=result,
            ),
            result_type=Activated,
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(seconds=rule.HEARTBEAT_TIMEOUT_S),
            retry_policy=_ACTIVATE_RETRY,
        )
        return WorkflowResult(
            job_id=prepared.job_id,
            outcome=result.outcome,
            version=activated.version or None,
        )
