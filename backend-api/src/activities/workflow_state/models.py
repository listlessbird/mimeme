from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from domain.image_ingest_input import ImageIngestInput
from shared.models import DuplicateReason, IngestStage, RebuildTrigger


class IngestUrlItem(BaseModel):
    id: int
    input: ImageIngestInput


class RecordIngestStageInput(BaseModel):
    ingest_url_id: int
    stage: IngestStage


class IngestInitOutput(BaseModel):
    urls: list[IngestUrlItem]


class MarkIngestUrlFailedInput(BaseModel):
    ingest_url_id: int
    error: str


class MarkIngestUrlDoneInput(BaseModel):
    ingest_url_id: int
    image_id: int
    duplicate_reason: DuplicateReason | None = None
    duplicate_of_image_id: int | None = None


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


class PrepareRebuildInput(BaseModel):
    job_id: str | None
    workflow_id: str
    force: bool
    trigger: RebuildTrigger


class PrepareRebuildOutput(BaseModel):
    decision: Literal["build", "clean", "busy"]
    job_id: str | None = None
    target_generation: int | None = None


class ReconcileGenerationInput(BaseModel):
    job_id: str
    target_generation: int


class ReleaseRebuildClaimInput(BaseModel):
    job_id: str


class StartRebuildJobInput(BaseModel):
    job_id: str


class FailRebuildJobInput(BaseModel):
    job_id: str
    error: str


class CompleteRebuildJobInput(BaseModel):
    job_id: str
    version: str
    num_vectors: int
    dimension: int
    removed_versions: list[str]
    text_num_vectors: int | None = None
