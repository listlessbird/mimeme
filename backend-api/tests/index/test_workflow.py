from __future__ import annotations

import uuid

from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from mimeme import index
from mimeme.index import rule
from mimeme.index.activity import Activities
from mimeme.index.workflow import RebuildWorkflow


def _build() -> index.Build:
    return index.Build(
        job_id="rebuild-1",
        version="v2-g3-test",
        target_generation=3,
        model="test/embed",
        index_type="flat",
        dimension=2,
        encoder=index.Encoder(repo="encoder", revision="rev", variant="model.onnx"),
        embeddings=[index.Embedding(image_id=1, image_key="embeddings/1.npy")],
    )


def test_exact_temporal_contract_names_and_poll_bounds() -> None:
    assert RebuildWorkflow.__temporal_workflow_definition.name == "mimeme.index.rebuild.v2"
    assert Activities.prepare.__temporal_activity_definition.name == "mimeme.index.prepare.v2"
    assert Activities.build.__temporal_activity_definition.name == "mimeme.index.build.v2"
    assert Activities.activate.__temporal_activity_definition.name == "mimeme.index.activate.v2"
    assert rule.workflow_id("abc") == "rebuild-index-v2-abc"
    assert rule.SCHEDULE_ID == "search-index-rebuild-v2"
    assert rule.POLL_INTERVAL_S <= 5
    assert rule.HEARTBEAT_TIMEOUT_S == 30


async def test_workflow_runs_three_coarse_retry_units_in_order() -> None:
    calls: list[str] = []
    build = _build()

    @activity.defn(name=rule.PREPARE_ACTIVITY)
    async def prepare(_: index.PrepareInput) -> index.Prepared:
        calls.append("prepare")
        return index.Prepared(decision="build", job_id=build.job_id, build=build)

    @activity.defn(name=rule.BUILD_ACTIVITY)
    async def run_build(_: index.Build) -> index.Result:
        calls.append("build")
        return index.Result(outcome="empty")

    @activity.defn(name=rule.ACTIVATE_ACTIVITY)
    async def activate(_: index.ActivateInput) -> index.Activated:
        calls.append("activate")
        return index.Activated(version="")

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        async with Worker(
            env.client,
            task_queue=rule.TASK_QUEUE,
            workflows=[RebuildWorkflow],
            activities=[prepare, run_build, activate],
        ):
            result = await env.client.execute_workflow(
                RebuildWorkflow.run,
                index.WorkflowInput(
                    job_id=build.job_id,
                    model=build.model,
                    index_type=build.index_type,
                    trigger=index.Trigger.MANUAL,
                ),
                id=str(uuid.uuid4()),
                task_queue=rule.TASK_QUEUE,
            )

    assert result.outcome == "empty"
    assert calls == ["prepare", "build", "activate"]


async def test_scheduled_busy_claim_exits_without_history_growth() -> None:
    seen: list[index.PrepareInput] = []

    @activity.defn(name=rule.PREPARE_ACTIVITY)
    async def prepare(input: index.PrepareInput) -> index.Prepared:
        seen.append(input)
        return index.Prepared(decision="busy")

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        async with Worker(
            env.client,
            task_queue=rule.TASK_QUEUE,
            workflows=[RebuildWorkflow],
            activities=[prepare],
        ):
            result = await env.client.execute_workflow(
                RebuildWorkflow.run,
                index.WorkflowInput(
                    job_id=None,
                    model="test/embed",
                    index_type="flat",
                    trigger=index.Trigger.SCHEDULED,
                ),
                id=str(uuid.uuid4()),
                task_queue=rule.TASK_QUEUE,
            )
    assert result == index.WorkflowResult(job_id=None, outcome="busy")
    assert seen[0].job_id is not None
    assert seen[0].job_id.startswith("rebuild-")
