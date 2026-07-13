from __future__ import annotations

import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELED"


class JobType(StrEnum):
    INGEST = "ingest"
    REBUILD_INDEX = "rebuild_index"


class ProcessingStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class DuplicateReason(StrEnum):
    SOURCE_SEEN = "SOURCE_SEEN"
    SHA256 = "SHA256"
    PHASH = "PHASH"


class IngestStage(StrEnum):
    QUEUED = "QUEUED"
    DOWNLOADING = "DOWNLOADING"
    PROCESSING = "PROCESSING"
    ANNOTATING = "ANNOTATING"
    EMBEDDING = "EMBEDDING"
    COMPLETE = "COMPLETE"
    DEDUPED = "DEDUPED"


class IngestInputKind(StrEnum):
    REMOTE_IMAGE_URL = "remote_image_url"
    STAGED_UPLOAD = "staged_upload"


class SourceType(StrEnum):
    API = "api"
    # `html` (scraping) is reserved for a future Adapter and intentionally not wired.


class SourceRunTrigger(StrEnum):
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class SourceRunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    type: Mapped[JobType] = mapped_column(SAEnum(JobType), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus), default=JobStatus.PENDING, nullable=False
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    workflow_id: Mapped[str | None] = mapped_column(String(256), nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingest_urls: Mapped[list[IngestURL]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class IngestURL(Base):
    __tablename__ = "ingest_urls"
    __table_args__ = (
        CheckConstraint(
            "(input_kind = 'remote_image_url' AND url IS NOT NULL AND artifact_key IS NULL) OR "
            "(input_kind = 'staged_upload' AND url IS NULL AND artifact_key IS NOT NULL)",
            name="ck_ingest_urls_input_payload",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    input_kind: Mapped[IngestInputKind] = mapped_column(
        SAEnum(IngestInputKind, values_callable=lambda enum: [member.value for member in enum]),
        default=IngestInputKind.REMOTE_IMAGE_URL,
        nullable=False,
    )
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_id: Mapped[int | None] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL"), nullable=True
    )

    # Pipeline position of this attempt (orthogonal to `status`). The workflow
    # advances it as it moves; on failure it stays frozen where it died.
    stage: Mapped[IngestStage] = mapped_column(
        SAEnum(IngestStage), default=IngestStage.QUEUED, nullable=False
    )
    stage_updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[Job] = relationship(back_populates="ingest_urls")
    image: Mapped[Image | None] = relationship(
        back_populates="ingest_urls", foreign_keys=[image_id]
    )

    duplicate_reason: Mapped[DuplicateReason | None] = mapped_column(
        SAEnum(DuplicateReason), nullable=True
    )

    duplicate_of_image_id: Mapped[int | None] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Provenance: which Source / run / item produced this URL. Manual uploads
    # leave all three null.
    source_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingestion_sources.id", ondelete="SET NULL"), nullable=True
    )
    source_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_runs.id", ondelete="SET NULL"), nullable=True
    )
    source_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_items.id", ondelete="SET NULL"), nullable=True
    )


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(Text, unique=True, nullable=False)

    dataset: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    original_filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    s3_etag: Mapped[str | None] = mapped_column(Text, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    format: Mapped[str | None] = mapped_column(String(10), nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    phash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(), server_default=func.now(), nullable=False
    )

    processing: Mapped[Processing | None] = relationship(
        back_populates="image", uselist=False, cascade="all, delete-orphan"
    )
    annotation: Mapped[Annotation | None] = relationship(
        back_populates="image", uselist=False, cascade="all, delete-orphan"
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )

    ingest_urls: Mapped[list[IngestURL]] = relationship(
        back_populates="image", foreign_keys="IngestURL.image_id"
    )


class Processing(Base):
    __tablename__ = "processing"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )

    ocr_status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    ocr_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_updated_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(), nullable=True)

    caption_status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    caption_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption_updated_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(), nullable=True)

    embed_status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    embed_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    embed_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embed_updated_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(), nullable=True)
    embed_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    image: Mapped[Image] = relationship(back_populates="processing")


class Annotation(Base):
    __tablename__ = "annotations"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)

    image: Mapped[Image] = relationship(back_populates="annotation")


class Artifact(Base):
    __tablename__ = "artifacts"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    kind: Mapped[str] = mapped_column(String(20), primary_key=True, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(), server_default=func.now(), nullable=False
    )

    image: Mapped[Image] = relationship(back_populates="artifacts")


class IndexBuild(Base):
    __tablename__ = "index_builds"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    embed_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    index_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    num_vectors: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=False, nullable=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(), server_default=func.now(), nullable=False
    )


class IngestionSource(Base):
    """A configured external origin of memes (e.g. a meme API + subreddit set).

    Soft delete uses `deleted_at`: a deleted Source keeps its row (and its
    `source_runs` / `source_items` for history) but is excluded from listings.

    Name uniqueness is enforced **only among live Sources** via a partial unique
    index (`WHERE deleted_at IS NULL`). Soft-deleting a Source frees its name, so
    delete-then-recreate with the same name just works. The cost: `name` is not
    globally unique (tombstones may share a name), so every lookup by name must
    carry the live predicate.
    """

    __tablename__ = "ingestion_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[SourceType] = mapped_column(
        SAEnum(SourceType), default=SourceType.API, nullable=False
    )
    adapter_key: Mapped[str] = mapped_column(String(100), nullable=False)
    adapter_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    dataset: Mapped[str | None] = mapped_column(String(255), nullable=True)
    schedule_cron: Mapped[str | None] = mapped_column(String(255), nullable=True)
    schedule_timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)
    max_items_per_run: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    runs: Mapped[list[SourceRun]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )
    items: Mapped[list[SourceItem]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "uq_ingestion_sources_name_live",
            "name",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )


class SourceRun(Base):
    """One execution of a Source — the higher-level tracking row above the
    INGEST `Job` it reuses (`ingest_job_id`). Status and counts are derived,
    never stored as live counters."""

    __tablename__ = "source_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("ingestion_sources.id"), nullable=False)
    trigger_mode: Mapped[SourceRunTrigger] = mapped_column(SAEnum(SourceRunTrigger), nullable=False)
    status: Mapped[SourceRunStatus] = mapped_column(
        SAEnum(SourceRunStatus), default=SourceRunStatus.PENDING, nullable=False
    )
    ingest_job_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    source: Mapped[IngestionSource] = relationship(back_populates="runs")

    __table_args__ = (
        Index(
            "ix_source_runs_source_id_created_at",
            "source_id",
            text("created_at DESC"),
        ),
        Index("ix_source_runs_status", "status"),
    )


class SourceItem(Base):
    """A distinct item discovered from a Source, deduped on
    `(source_id, external_item_id)`. `last_source_run_id` is a plain
    "last touched by" pointer, not a strong ownership edge."""

    __tablename__ = "source_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("ingestion_sources.id"), nullable=False)
    last_source_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    external_item_id: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_item_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    canonical_image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    first_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    source: Mapped[IngestionSource] = relationship(back_populates="items")

    __table_args__ = (
        UniqueConstraint("source_id", "external_item_id", name="uq_source_items_source_external"),
        Index("ix_source_items_source_id", "source_id"),
    )
