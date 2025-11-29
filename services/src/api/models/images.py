from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class ImageIngestRequest(BaseModel):
    urls: list[HttpUrl] = Field(
        ..., min_length=1, max_length=100, description="List of image urls to ingest"
    )
    tags: list[str] = Field(default_factory=list, description="tags to apply")
    priority: Literal["low", "normal", "high"] = Field(
        default="normal",
        description="Processing priority",
    )
    callback_url: HttpUrl | None = Field(
        None,
        description="Webhook URL to call when processing completes",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "urls": [
                    "https://example.com/meme1.jpg",
                    "https://example.com/meme2.png",
                ],
                "tags": ["funny", "cats"],
                "priority": "normal",
            }
        }
    }


class ImageIngestResponse(BaseModel):
    job_id: str = Field(..., description="job id for tracking progress")
    queued: int = Field(..., description=" Number of images queued")
    duplicates: int = Field(default=0, description="Number of duplicate images")
    message: str = Field(..., description="Status Message")


class ImageStatus(str, Enum):
    """Image processing status."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    SCANNING = "scanning"
    ANNOTATING = "annotating"
    EMBEDDING = "embedding"
    DONE = "done"
    FAILED = "failed"


class ImageResponse(BaseModel):
    id: int
    sha256: str
    rel_path: str
    url: str | None = None
    s3_key: str | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None
    phash: str | None = None

    status: ImageStatus
    ocr_status: str | None = None
    caption_status: str | None = None
    embed_status: str | None = None

    caption: str | None = None
    ocr_text: str | None = None
    tags: list[str] = Field(default_factory=list)

    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class ImageListResponse(BaseModel):
    images: list[ImageResponse]
    total: int
    limit: int
    offset: int
    has_more: bool
