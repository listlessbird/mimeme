from workflows.ingest import IngestWorkflow
from workflows.models import (
    IngestWorkflowInput,
    IngestWorkflowOutput,
    RebuildIndexWorkflowInput,
    RebuildIndexWorkflowOutput,
)
from workflows.rebuild_index import RebuildIndexWorkflow

ALL_WORKFLOWS = [
    IngestWorkflow,
    RebuildIndexWorkflow,
]

__all__ = [
    "ALL_WORKFLOWS",
    "IngestWorkflow",
    "IngestWorkflowInput",
    "IngestWorkflowOutput",
    "RebuildIndexWorkflow",
    "RebuildIndexWorkflowInput",
    "RebuildIndexWorkflowOutput",
]
