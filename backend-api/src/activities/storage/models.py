from __future__ import annotations

from pydantic import BaseModel


class DownloadImageInput(BaseModel):
    url: str
    job_id: str
    ingest_url_id: int


class DownloadImageOutput(BaseModel):
    ingest_url_id: int
    local_path: str
    filename: str
    success: bool
    error: str | None = None


class ProcessImageInput(BaseModel):
    local_path: str
    filename: str
    ingest_url_id: int
    dataset: str | None = None


class ProcessImageOutput(BaseModel):
    ingest_url_id: int
    image_id: int
    sha256: str
    s3_key: str
    width: int | None
    height: int | None
    format: str | None
    is_duplicate: bool = False


class ImageMetadata(BaseModel):
    sha256: str
    phash: str | None
    width: int | None
    height: int | None
    format: str | None
    filesize: int
