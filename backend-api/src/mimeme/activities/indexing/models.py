from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BuildIndexInput(BaseModel):
    model_name: str
    index_type: str = Field(default="flat")
    force: bool = Field(default=False)
    target_generation: int


class BuildIndexOutput(BaseModel):
    outcome: Literal["built", "empty_reconcile"] = "built"
    version: str | None = None
    num_vectors: int = 0
    dimension: int | None = None
    s3_key: str | None = None
    text_num_vectors: int | None = None
    text_s3_key: str | None = None


class SwapIndexInput(BaseModel):
    version: str
    job_id: str
    target_generation: int


class GarbageCollectOutput(BaseModel):
    removed_versions: list[str]
