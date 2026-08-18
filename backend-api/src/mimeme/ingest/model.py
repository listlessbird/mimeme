from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mimeme.db.schema import DuplicateReason, IngestInputKind

MAX_SUBMISSION_URLS = 100


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RemoteUrl(_Frozen):
    kind: Literal["remote_image_url"] = "remote_image_url"
    url: str = Field(min_length=1)


class Staged(_Frozen):
    kind: Literal["staged_upload"] = "staged_upload"
    artifact_key: str = Field(min_length=1)


Source = Annotated[RemoteUrl | Staged, Field(discriminator="kind")]


def restore(*, kind: IngestInputKind, url: str | None, artifact_key: str | None) -> Source:
    match kind:
        case IngestInputKind.REMOTE_IMAGE_URL:
            if url is None:
                raise ValueError("remote image input is missing its URL")
            return RemoteUrl(url=url)
        case IngestInputKind.STAGED_UPLOAD:
            if artifact_key is None:
                raise ValueError("staged upload input is missing its artifact key")
            return Staged(artifact_key=artifact_key)


class Input(_Frozen):
    job_id: str
    item_id: int
    source: Source
    dataset: str | None = None


class ItemRef(_Frozen):
    item_id: int
    source: Source


class Submission(_Frozen):
    urls: list[Source] = Field(min_length=1, max_length=MAX_SUBMISSION_URLS)
    dataset: str | None = None
    tags: list[str] = Field(default_factory=list)
    callback_url: str | None = None


Outcome = Literal["processed", "duplicate", "failed"]


class Result(_Frozen):
    item_id: int
    outcome: Outcome
    image_id: int | None = None
    duplicate_reason: DuplicateReason | None = None
    error: str | None = None
    download_ms: float | None = Field(default=None, ge=0)
    annotation_ms: float | None = Field(default=None, ge=0)
    embedding_ms: float | None = Field(default=None, ge=0)
    total_ms: float | None = Field(default=None, ge=0)


class WorkflowInput(_Frozen):
    job_id: str
    dataset: str | None
    items: list[ItemRef]


class Finish(_Frozen):
    job_id: str


class Error(Exception):
    pass


class InvalidImage(Error):
    """Terminal per-image failure. The item is marked failed; no retry."""


class Retryable(Error):
    """Transient infrastructure failure. The activity should retry."""
