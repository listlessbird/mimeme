from typing import Literal

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    id: int = Field(..., description="Image Id")
    sha256: str = Field(..., description="Image hash")
    score: float = Field(..., description="Similarity Score")
    url: str | None = Field(None, description="Image url")
    rel_path: str = Field(..., description="Relative path to img")
    caption: str | None = Field(None, description="Generated Caption")
    ocr_text: str | None = Field(None, description="Extracted text from the image")
    width: int | None = Field(None, description="Image width")
    height: int | None = Field(None, description="Image height")


class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=500, description="Search query text")
    limit: int = Field(default=20, ge=1, le=100, description="Number of results to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    mode: Literal["image", "text", "hybrid"] = Field(
        default="hybrid",
        description="Search mode: image (visual similarity), text (caption/OCR), hybrid (both)",
    )


class SearchResponse(BaseModel):
    query: str = Field(..., description="Original query")
    results: list[SearchResult] = Field(default_factory=list, description="Search results")
    total: int = Field(..., description="Total number of matching results")
    limit: int = Field(..., description="Requested limit")
    offset: int = Field(..., description="Requested offset")
    search_time_ms: float = Field(..., description="Search time in milliseconds")
    index_version: str | None = Field(None, description="Active index version")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "cats",
                "results": [
                    {
                        "id": 1,
                        "sha256": "abc123...",
                        "score": 0.95,
                        "url": "https://s3.example.com/memes/ab/c1/abc123.jpg",
                        "rel_path": "funny-cats/cat1.jpg",
                        "caption": "A cat sleeping on a keyboard",
                        "ocr_text": None,
                        "width": 800,
                        "height": 600,
                    }
                ],
                "total": 150,
                "limit": 20,
                "offset": 0,
                "search_time_ms": 12.5,
                "index_version": "v20251125-001",
            }
        }
    }
