from __future__ import annotations

from pydantic import BaseModel, Field
from sympy.functions.special.tests.test_error_functions import w

from shared.models.orm import SourceRunTrigger


class IngestWorkflowInput(BaseModel):
    job_id: str
    dataset: str | None = None
    tags: list[str] = Field(default_factory=list)
    callback_url: str | None = None


class IngestWorkflowOutput(BaseModel):
    job_id: str
    total: int
    processed: int
    failed: int
    duplicates: int


class RebuildIndexWorkflowInput(BaseModel):
    job_id: str
    force: bool = False
    model_name: str | None = None


class RebuildIndexWorkflowOutput(BaseModel):
    job_id: str
    version: str
    num_vectors: int
    dimension: int
    removed_versions: list[str]
    text_num_vectors: int | None = None


class SourceSyncWorkflowInput(BaseModel):
    source_id: int
    trigger_mode: SourceRunTrigger = SourceRunTrigger.MANUAL


class SourceSyncWorkflowOutput(BaseModel):
    source_id: int
    source_run_id: int | None = None
    discovered: int = 0
    seen: int = 0
    queued: int = 0
    duplicates: int = 0
    failed: int = 0
    skipped_unsupported: int = 0
    skipped_invalid: int = 0
