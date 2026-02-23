from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="Service health state")


class IndexVersionResponse(BaseModel):
    version: int
    embed_model: str
    index_type: str
    num_vectors: int
    dimension: int
    is_active: bool
    created_at: str | None = None


class IndexVersionsResponse(BaseModel):
    versions: list[IndexVersionResponse]
