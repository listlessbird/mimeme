from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mimeme.db.schema import (
    DuplicateReason,
    IngestStage,
    JobStatus,
    JobType,
    RebuildTrigger,
)
from mimeme.ingest.model import Source

JOB_ERROR_LIMIT = 2000
INGEST_URL_ERROR_LIMIT = 1000


class NotFound(Exception):
    pass


class InvalidState(Exception):
    pass


class ClaimOwnership(Exception):
    pass


class ClaimTarget(Exception):
    pass


class StateMissing(Exception):
    pass


class IngestResult(BaseModel, frozen=True, extra="forbid"):
    processed: int = Field(ge=0)
    failed: int = Field(ge=0)
    duplicates: int = Field(ge=0)


class RebuildResult(BaseModel, frozen=True, extra="forbid"):
    version: str
    num_vectors: int = Field(ge=0)
    dimension: int = Field(ge=0)
    removed_versions: list[str]
    text_num_vectors: int | None = Field(default=None, ge=0)
    skipped: bool = False
    skip_reason: str | None = None


class RawResult(BaseModel, frozen=True, extra="forbid"):
    raw: str


type Result = IngestResult | RebuildResult | RawResult


class View(BaseModel, frozen=True, extra="forbid"):
    id: str
    type: JobType
    status: JobStatus
    progress: float
    message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    result: Result | None


class Page(BaseModel, frozen=True, extra="forbid"):
    jobs: list[View]
    total: int


class IngestCreation(BaseModel, frozen=True, extra="forbid"):
    job_id: str
    workflow_id: str
    queued: int
    duplicates: int
    dataset: str | None
    tags: list[str]
    callback_url: str | None


class RebuildCreation(BaseModel, frozen=True, extra="forbid"):
    job: View
    workflow_id: str
    force: bool
    model_name: str
    index_type: Literal["flat", "hnsw"]


class Cancellation(BaseModel, frozen=True, extra="forbid"):
    workflow_id: str | None


class BuildView(BaseModel, frozen=True, from_attributes=True, extra="forbid"):
    version: str
    embed_model: str | None
    index_type: str | None
    num_vectors: int | None
    dimension: int | None
    is_active: bool
    created_at: datetime | None


class RowData(BaseModel, frozen=True, from_attributes=True, extra="forbid"):
    id: str
    type: JobType
    status: JobStatus
    progress: float
    message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    result: str | None


class SourceIngestItem(BaseModel, frozen=True, extra="forbid"):
    url: str
    source_id: int
    source_run_id: int
    source_item_id: int


class IngestUrlRef(BaseModel, frozen=True, extra="forbid"):
    id: int
    input: Source


class IngestInit(BaseModel, frozen=True, extra="forbid"):
    urls: list[IngestUrlRef]


class ItemDone(BaseModel, frozen=True, extra="forbid"):
    found: bool
    image_exists: bool | None


class Claim(BaseModel, frozen=True, extra="forbid"):
    job_id: str
    target_generation: int
    claimed_at: datetime


class Freshness(BaseModel, frozen=True, extra="forbid"):
    desired_generation: int
    active_generation: int
    is_stale: bool
    rebuild_job_id: str | None
    rebuild_target_generation: int | None
    rebuild_claimed_at: datetime | None
    last_dirty_at: datetime | None
    last_dirty_reason: str | None
    last_reconciled_at: datetime | None

    @property
    def active_claim(self) -> Claim | None:
        if (
            self.rebuild_job_id is None
            or self.rebuild_target_generation is None
            or self.rebuild_claimed_at is None
        ):
            return None
        return Claim(
            job_id=self.rebuild_job_id,
            target_generation=self.rebuild_target_generation,
            claimed_at=self.rebuild_claimed_at,
        )


class FreshnessStatus(BaseModel, frozen=True, extra="forbid"):
    view: Freshness
    active_version: str | None


class ClaimResult(BaseModel, frozen=True, extra="forbid"):
    acquired: bool
    reason: Literal["acquired", "clean", "busy"]
    view: Freshness


class PrepareDecision(BaseModel, frozen=True, extra="forbid"):
    decision: Literal["build", "clean", "busy"]
    job_id: str | None = None
    target_generation: int | None = None


class EmbeddingSaved(BaseModel, frozen=True, extra="forbid"):
    found: bool
    index_changed: bool
    desired_generation: int | None


class AnnotationSave(BaseModel, frozen=True, extra="forbid"):
    image_id: int
    caption: str
    caption_model: str
    ocr_text: str
    ocr_model: str


class EmbeddingSave(BaseModel, frozen=True, extra="forbid"):
    image_id: int
    model: str
    dimension: int
    image_embedding_key: str


class InitializeCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["initialize"] = "initialize"
    job_id: str


class StageCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["stage"] = "stage"
    ingest_url_id: int
    stage: IngestStage


class FailItemCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["fail_item"] = "fail_item"
    ingest_url_id: int
    error: str


class CompleteItemCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["complete_item"] = "complete_item"
    ingest_url_id: int
    image_id: int
    duplicate_reason: DuplicateReason | None = None
    duplicate_of_image_id: int | None = None
    similar_image_id: int | None = None
    phash_distance: int | None = None


class SaveInferenceCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["save_inference"] = "save_inference"
    annotation: AnnotationSave | None = None
    embedding: EmbeddingSave | None = None


class ProgressCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["progress"] = "progress"
    job_id: str
    progress: float
    message: str | None = None


class CompleteIngestCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["complete"] = "complete"
    job_id: str
    processed: int
    failed: int
    duplicates: int


type IngestStateCommand = Annotated[
    InitializeCommand
    | StageCommand
    | FailItemCommand
    | CompleteItemCommand
    | SaveInferenceCommand
    | ProgressCommand
    | CompleteIngestCommand,
    Field(discriminator="op"),
]


class IngestStateOutput(BaseModel, frozen=True, extra="forbid"):
    init: IngestInit | None = None
    found: bool | None = None
    image_exists: bool | None = None
    index_changed: bool | None = None
    desired_generation: int | None = None


class PrepareCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["prepare"] = "prepare"
    job_id: str | None
    workflow_id: str
    force: bool
    trigger: RebuildTrigger


class ReconcileCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["reconcile"] = "reconcile"
    job_id: str
    target_generation: int


class ReleaseCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["release"] = "release"
    job_id: str


class StartCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["start"] = "start"
    job_id: str


class FailRebuildCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["fail"] = "fail"
    job_id: str
    error: str


class CompleteRebuildCommand(BaseModel, frozen=True, extra="forbid"):
    op: Literal["complete"] = "complete"
    job_id: str
    version: str
    num_vectors: int
    dimension: int
    removed_versions: list[str]
    text_num_vectors: int | None = None


type RebuildStateCommand = Annotated[
    PrepareCommand
    | ReconcileCommand
    | ReleaseCommand
    | StartCommand
    | FailRebuildCommand
    | CompleteRebuildCommand,
    Field(discriminator="op"),
]


class RebuildStateOutput(BaseModel, frozen=True, extra="forbid"):
    decision: PrepareDecision | None = None
    released: bool | None = None
