"""Integration tests for SourceRetryWorkflow using Temporal's test environment.

The retry workflow reprocesses an already-prepared INGEST job (the failed URLs
were reset + reattached to it by the request handler) via the child
IngestWorkflow, then re-finalizes each affected Source run so its status is
re-derived from the fresh attempt outcomes.

The child IngestWorkflow and the finalize activity are mocked by name; the real
SourceRetryWorkflow runs.
"""

from __future__ import annotations

import uuid

import pytest
from temporalio import activity, workflow
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from mimeme.activities.ingestion.models import FinalizeSourceRunInput, FinalizeSourceRunOutput
from mimeme.db.schema import SourceRunStatus
from mimeme.workflows.models import (
    IngestWorkflowInput,
    IngestWorkflowOutput,
    SourceRetryWorkflowInput,
    SourceRetryWorkflowOutput,
)
from mimeme.workflows.source_retry import SourceRetryWorkflow

_finalized: list[int] = []
_ingested: list[str] = []


@activity.defn(name="finalize_source_run_activity")
async def mock_finalize(input: FinalizeSourceRunInput) -> FinalizeSourceRunOutput:
    _finalized.append(input.source_run_id)
    return FinalizeSourceRunOutput(
        status=SourceRunStatus.COMPLETED, discovered=1, queued=1, duplicate=0, failed=0
    )


@workflow.defn(name="IngestWorkflow", sandboxed=False)
class MockIngestWorkflow:
    @workflow.run
    async def run(self, input: IngestWorkflowInput) -> IngestWorkflowOutput:
        _ingested.append(input.job_id)
        return IngestWorkflowOutput(
            job_id=input.job_id, total=1, processed=1, failed=0, duplicates=0
        )


@pytest.fixture(autouse=True)
def _reset() -> None:
    _finalized.clear()
    _ingested.clear()


async def _run(input: SourceRetryWorkflowInput) -> SourceRetryWorkflowOutput:
    task_queue = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[SourceRetryWorkflow, MockIngestWorkflow],
            activities=[mock_finalize],
        ):
            return await env.client.execute_workflow(
                SourceRetryWorkflow.run,
                input,
                id=str(uuid.uuid4()),
                task_queue=task_queue,
            )


class TestSourceRetryWorkflow:
    async def test_reingests_then_finalizes_each_affected_run(self) -> None:
        result = await _run(
            SourceRetryWorkflowInput(
                job_id="ingest-retry-1", source_run_ids=[7, 9], dataset="memes"
            )
        )

        assert _ingested == ["ingest-retry-1"]
        assert _finalized == [7, 9]
        assert result.job_id == "ingest-retry-1"
        assert result.statuses == [SourceRunStatus.COMPLETED, SourceRunStatus.COMPLETED]
