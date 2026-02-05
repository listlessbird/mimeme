from __future__ import annotations

from pydantic import BaseModel, Field


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
