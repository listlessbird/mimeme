from workflows.ingest import IngestWorkflow
from workflows.models import (
    IngestWorkflowInput,
    IngestWorkflowOutput,
    RebuildIndexWorkflowInput,
    RebuildIndexWorkflowOutput,
    SourceSyncWorkflowInput,
    SourceSyncWorkflowOutput,
)
from workflows.rebuild_index import RebuildIndexWorkflow
from workflows.source_sync import SourceSyncWorkflow

ALL_WORKFLOWS = [IngestWorkflow, RebuildIndexWorkflow, SourceSyncWorkflow]

__all__ = [
    "ALL_WORKFLOWS",
    "IngestWorkflow",
    "IngestWorkflowInput",
    "IngestWorkflowOutput",
    "RebuildIndexWorkflow",
    "RebuildIndexWorkflowInput",
    "RebuildIndexWorkflowOutput",
    "SourceSyncWorkflow",
    "SourceSyncWorkflowInput",
    "SourceSyncWorkflowOutput",
]
