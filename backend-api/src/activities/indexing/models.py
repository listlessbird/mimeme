from __future__ import annotations

from pydantic import BaseModel, Field


class BuildIndexInput(BaseModel):
    model_name: str
    index_type: str = Field(default="flat")
    force: bool = Field(default=False)


class BuildIndexOutput(BaseModel):
    version: str
    num_vectors: int
    dimension: int
    s3_key: str
    text_num_vectors: int | None = None
    text_s3_key: str | None = None


class SwapIndexInput(BaseModel):
    version: str


class GarbageCollectOutput(BaseModel):
    removed_versions: list[str]
