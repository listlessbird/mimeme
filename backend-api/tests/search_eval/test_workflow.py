from __future__ import annotations

import uuid

from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from mimeme import search
from mimeme.search_eval import rule
from mimeme.search_eval.activity import Activities
from mimeme.search_eval.model import (
    FailureInput,
    PreparedRun,
    QuerySpec,
    RetrievalBatch,
    WorkflowInput,
    WorkflowResult,
)
from mimeme.search_eval.workflow import SearchEvalScoreWorkflow, SearchEvalWorkflow

_TASK_QUEUE = "search-eval-tests"


def test_exact_temporal_contract_names() -> None:
    assert SearchEvalWorkflow.__temporal_workflow_definition.name == "mimeme.search_eval.run.v1"
    assert (
        SearchEvalScoreWorkflow.__temporal_workflow_definition.name == "mimeme.search_eval.score.v1"
    )
    assert Activities.prepare.__temporal_activity_definition.name == rule.PREPARE_ACTIVITY
    assert Activities.retrieve.__temporal_activity_definition.name == rule.RETRIEVE_ACTIVITY
    assert Activities.score.__temporal_activity_definition.name == rule.SCORE_ACTIVITY
    assert rule.run_workflow_id("abc") == "search-eval-v1-abc"


async def test_run_workflow_retrieves_sequential_batches_then_scores() -> None:
    calls: list[str] = []
    queries = [QuerySpec(id=index, text=f"query {index}") for index in range(26)]

    @activity.defn(name=rule.PREPARE_ACTIVITY)
    async def prepare(input: WorkflowInput) -> PreparedRun:
        calls.append("prepare")
        return PreparedRun(
            run_id=input.run_id,
            recipe_id="image_siglip_text",
            recipe=search.recipe.resolve("image_siglip_text"),
            index_version="index-v1",
            queries=queries,
        )

    @activity.defn(name=rule.RETRIEVE_ACTIVITY)
    async def retrieve(input: RetrievalBatch) -> None:
        calls.append(f"retrieve:{len(input.queries)}")

    @activity.defn(name=rule.SCORE_ACTIVITY)
    async def score(input: WorkflowInput) -> WorkflowResult:
        calls.append("score")
        return WorkflowResult(run_id=input.run_id, status="complete")

    @activity.defn(name=rule.FAIL_ACTIVITY)
    async def fail(input: FailureInput) -> WorkflowResult:
        calls.append("fail")
        return WorkflowResult(run_id=input.run_id, status="failed")

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        async with Worker(
            env.client,
            task_queue=_TASK_QUEUE,
            workflows=[SearchEvalWorkflow],
            activities=[prepare, retrieve, score, fail],
        ):
            result = await env.client.execute_workflow(
                SearchEvalWorkflow.run,
                WorkflowInput(run_id="run-1"),
                id=str(uuid.uuid4()),
                task_queue=_TASK_QUEUE,
            )

    assert result == WorkflowResult(run_id="run-1", status="complete")
    assert calls == ["prepare", "retrieve:25", "retrieve:1", "score"]
