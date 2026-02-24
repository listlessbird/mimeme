from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="Service health state")


class IndexVersionResponse(BaseModel):
    version: str
    embed_model: str | None = None
    index_type: str | None = None
    num_vectors: int | None = None
    dimension: int | None = None
    is_active: bool
    created_at: str | None = None


class IndexVersionsResponse(BaseModel):
    versions: list[IndexVersionResponse]
