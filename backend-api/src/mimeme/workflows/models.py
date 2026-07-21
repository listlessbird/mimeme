from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mimeme.db.schema import RebuildTrigger, SourceRunStatus, SourceRunTrigger


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
    job_id: str | None = None
    force: bool = False
    model_name: str
    index_type: str
    trigger: RebuildTrigger = RebuildTrigger.MANUAL


class RebuildIndexWorkflowOutput(BaseModel):
    job_id: str | None
    outcome: Literal["built", "empty_reconcile", "skipped", "busy"]
    version: str | None = None
    num_vectors: int = 0
    dimension: int | None = None
    removed_versions: list[str] = []
    text_num_vectors: int | None = None


class SourceSyncWorkflowInput(BaseModel):
    source_id: int
    trigger: SourceRunTrigger = SourceRunTrigger.MANUAL


class SourceSyncWorkflowOutput(BaseModel):
    source_run_id: int
    status: SourceRunStatus
    discovered: int
    queued: int
    duplicate: int
    failed: int
    ingest_job_id: str | None = None


class SourceRetryWorkflowInput(BaseModel):
    job_id: str
    source_run_ids: list[int]
    dataset: str | None = None


class SourceRetryWorkflowOutput(BaseModel):
    job_id: str
    source_run_ids: list[int]
    statuses: list[SourceRunStatus]
