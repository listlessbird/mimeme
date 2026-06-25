from workflows.ingest import IngestWorkflow
from workflows.models import (
    IngestWorkflowInput,
    IngestWorkflowOutput,
    RebuildIndexWorkflowInput,
    RebuildIndexWorkflowOutput,
    SourceRetryWorkflowInput,
    SourceRetryWorkflowOutput,
    SourceSyncWorkflowInput,
    SourceSyncWorkflowOutput,
)
from workflows.rebuild_index import RebuildIndexWorkflow
from workflows.source_retry import SourceRetryWorkflow
from workflows.source_sync import SourceSyncWorkflow

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
