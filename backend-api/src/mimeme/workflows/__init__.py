from mimeme.workflows.ingest import IngestWorkflow
from mimeme.workflows.models import (
    IngestWorkflowInput,
    IngestWorkflowOutput,
    RebuildIndexWorkflowInput,
    RebuildIndexWorkflowOutput,
    SourceRetryWorkflowInput,
    SourceRetryWorkflowOutput,
    SourceSyncWorkflowInput,
    SourceSyncWorkflowOutput,
)
from mimeme.workflows.rebuild_index import RebuildIndexWorkflow
from mimeme.workflows.source_retry import SourceRetryWorkflow
from mimeme.workflows.source_sync import SourceSyncWorkflow

ALL_WORKFLOWS = [
    IngestWorkflow,
    RebuildIndexWorkflow,
    SourceSyncWorkflow,
    SourceRetryWorkflow,
]

__all__ = [
    "ALL_WORKFLOWS",
    "IngestWorkflow",
    "IngestWorkflowInput",
    "IngestWorkflowOutput",
    "RebuildIndexWorkflow",
    "RebuildIndexWorkflowInput",
    "RebuildIndexWorkflowOutput",
    "SourceRetryWorkflow",
    "SourceRetryWorkflowInput",
    "SourceRetryWorkflowOutput",
    "SourceSyncWorkflow",
    "SourceSyncWorkflowInput",
    "SourceSyncWorkflowOutput",
]
