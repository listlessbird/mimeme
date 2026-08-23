from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from mimeme.search_eval import rule
    from mimeme.search_eval.model import (
        FailureInput,
        PreparedRun,
        RetrievalBatch,
        WorkflowInput,
        WorkflowResult,
    )

_RETRY = RetryPolicy(maximum_attempts=rule.MAX_ATTEMPTS, maximum_interval=timedelta(seconds=30))


async def _fail(input: WorkflowInput, failure: Exception) -> WorkflowResult:
    return await workflow.execute_activity(
        rule.FAIL_ACTIVITY,
        FailureInput(run_id=input.run_id, error=str(failure)),
        result_type=WorkflowResult,
        start_to_close_timeout=timedelta(minutes=1),
        retry_policy=_RETRY,
    )


@workflow.defn(name=rule.RUN_WORKFLOW)
class SearchEvalWorkflow:
    @workflow.run
    async def run(self, input: WorkflowInput) -> WorkflowResult:
        try:
            prepared: PreparedRun = await workflow.execute_activity(
                rule.PREPARE_ACTIVITY,
                input,
                result_type=PreparedRun,
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=_RETRY,
            )
            for offset in range(0, len(prepared.queries), rule.BATCH_SIZE):
                await workflow.execute_activity(
                    rule.RETRIEVE_ACTIVITY,
                    RetrievalBatch(
                        run_id=prepared.run_id,
                        mode=prepared.mode,
                        index_version=prepared.index_version,
                        queries=prepared.queries[offset : offset + rule.BATCH_SIZE],
                    ),
                    start_to_close_timeout=timedelta(minutes=45),
                    heartbeat_timeout=timedelta(seconds=rule.HEARTBEAT_TIMEOUT_S),
                    retry_policy=_RETRY,
                )
            return await workflow.execute_activity(
                rule.SCORE_ACTIVITY,
                input,
                result_type=WorkflowResult,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=_RETRY,
            )
        except Exception as failure:
            return await _fail(input, failure)


@workflow.defn(name=rule.SCORE_WORKFLOW)
class SearchEvalScoreWorkflow:
    @workflow.run
    async def run(self, input: WorkflowInput) -> WorkflowResult:
        try:
            return await workflow.execute_activity(
                rule.SCORE_ACTIVITY,
                input,
                result_type=WorkflowResult,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=_RETRY,
            )
        except Exception as failure:
            return await _fail(input, failure)
