from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class IngestFailureDecision(BaseModel, frozen=True):
    error: str


class IngestDoneDecision(BaseModel, frozen=True):
    image_id: int


class IngestEmbeddingDecision(BaseModel, frozen=True):
    image_id: int
    model: str
    dimension: int
    image_embedding_key: str


class IngestProgress(BaseModel, frozen=True):
    processed: int
    failed: int
    duplicates: int
    total: int
    progress: float


class IngestCompletion(BaseModel, frozen=True):
    job_id: str
    total: int
    processed: int
    failed: int
    duplicates: int


class IngestPolicy:
    def __init__(
        self,
        *,
        total: int,
        processed: int = 0,
        failed: int = 0,
        duplicates: int = 0,
    ) -> None:
        self.total = total
        self.processed = processed
        self.failed = failed
        self.duplicates = duplicates

    def record_download_failure(self, error: str | None) -> IngestFailureDecision:
        self.failed += 1
        message = (error or "Download failed").replace("\n", " ").strip()
        return IngestFailureDecision(
            error=message if len(message) <= 500 else message[:497] + "..."
        )

    def record_duplicate(self, image_id: int) -> IngestDoneDecision:
        self.duplicates += 1
        return IngestDoneDecision(image_id=image_id)

    def record_success(self, image_id: int) -> IngestDoneDecision:
        self.processed += 1
        return IngestDoneDecision(image_id=image_id)

    def record_exception(self, exc: Exception) -> IngestFailureDecision:
        self.failed += 1
        message = str(exc).replace("\n", " ").strip()
        return IngestFailureDecision(
            error=message if len(message) <= 500 else message[:497] + "..."
        )

    def progress_state(self) -> IngestProgress:
        completed = self.processed + self.failed + self.duplicates
        progress = (completed / self.total) * 100 if self.total > 0 else 0
        return IngestProgress(
            processed=self.processed,
            failed=self.failed,
            duplicates=self.duplicates,
            total=self.total,
            progress=progress,
        )

    def completion(self, job_id: str) -> IngestCompletion:
        return IngestCompletion(
            job_id=job_id,
            total=self.total,
            processed=self.processed,
            failed=self.failed,
            duplicates=self.duplicates,
        )

    def compose_embedding_text(self, caption: str | None, ocr_text: str | None) -> str:
        return " ".join(part for part in [caption, ocr_text] if part)

    def embedding_decision(self, results: list[Any]) -> IngestEmbeddingDecision | None:
        if not results:
            return None
        result = results[0]
        return IngestEmbeddingDecision(
            image_id=result.image_id,
            model=result.model,
            dimension=result.dimension,
            image_embedding_key=result.image_embedding_key,
        )
