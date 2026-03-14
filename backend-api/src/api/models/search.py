from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    id: int = Field(description="Image ID")
    sha256: str = Field(description="Image hash")
    score: float = Field(description="Similarity score")
    url: str | None = Field(default=None, description="Image URL")
    caption: str | None = Field(default=None, description="Generated caption")
    ocr_text: str | None = Field(default=None, description="Extracted text from image")
    width: int | None = Field(default=None, description="Image width")
    height: int | None = Field(default=None, description="Image height")


class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=500, description="Search query text")
    limit: int = Field(default=20, ge=1, le=100, description="Number of results")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    mode: Literal["hybrid"] | None = Field(
        default=None,
        description="Optional mode. Omit for image search; use 'hybrid' to fuse image and text indexes.",
    )


class SearchResponse(BaseModel):
    query: str = Field(description="Original query")
    results: list[SearchResult] = Field(default_factory=list, description="Search results")
    total: int = Field(description="Total matching results")
    limit: int
    offset: int
    search_time_ms: float = Field(description="Search time in milliseconds")
    index_version: str | None = Field(default=None, description="Active index version")
