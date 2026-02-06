from __future__ import annotations

from pydantic import BaseModel


class IngestUrlItem(BaseModel):
    id: int
    url: str


class IngestInitOutput(BaseModel):
    urls: list[IngestUrlItem]


class MarkIngestUrlFailedInput(BaseModel):
    ingest_url_id: int
    error: str


class MarkIngestUrlDoneInput(BaseModel):
    ingest_url_id: int
    image_id: int


class SaveAnnotationsInput(BaseModel):
    image_id: int
    caption: str
    caption_model: str
    ocr_text: str
    ocr_model: str


class SaveEmbeddingInfoInput(BaseModel):
    image_id: int
    model: str
    dimension: int
    image_embedding_key: str


class UpdateJobProgressInput(BaseModel):
    job_id: str
    progress: float
    message: str | None = None


class CompleteIngestJobInput(BaseModel):
    job_id: str
    processed: int
    failed: int
    duplicates: int


class StartRebuildJobInput(BaseModel):
    job_id: str


class CompleteRebuildJobInput(BaseModel):
    job_id: str
    version: str
    num_vectors: int
    dimension: int
    removed_versions: list[str]
