import datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column, relationship


class Base(MappedAsDataclass, DeclarativeBase):
    pass


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELED"


class JobType(str, Enum):
    INGEST = "ingest"
    REBUILD_INDEX = "rebuild_index"


class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class SourceType(str, Enum):
    API = "api"
    HTML = "html"


class SourceRunTrigger(str, Enum):
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class SourceRunStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class DuplicateReason(str, Enum):
    SHA256 = "SHA256"
    PHASH = "PHASH"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    type: Mapped[JobType] = mapped_column(SAEnum(JobType), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus), default=JobStatus.PENDING, nullable=False
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    result: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    workflow_id: Mapped[str | None] = mapped_column(String(256), nullable=True, default=None)
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, init=False
    )
    ingest_urls: Mapped[list["IngestURL"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", default_factory=list, init=False
    )


class IngestURL(Base):
    __tablename__ = "ingest_urls"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    image_id: Mapped[int | None] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL"), nullable=True, default=None
    )
    source_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingestion_sources.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        default=None,
    )
    source_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    source_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_items.id", ondelete="SET NULL"), nullable=True, index=True, default=None
    )
    duplicate_reason: Mapped[DuplicateReason | None] = mapped_column(
        SAEnum(DuplicateReason), nullable=True, default=None
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, init=False
    )
    job: Mapped["Job"] = relationship(back_populates="ingest_urls", init=False)
    image: Mapped["Image | None"] = relationship(
        back_populates="ingest_urls", default=None, init=False
    )
    source: Mapped["IngestionSource | None"] = relationship(default=None, init=False)
    source_run: Mapped["SourceRun | None"] = relationship(default=None, init=False)
    source_item: Mapped["SourceItem | None"] = relationship(default=None, init=False)


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    sha256: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    dataset: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True, default=None
    )
    original_filename: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    s3_etag: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    format: Mapped[str | None] = mapped_column(String(10), nullable=True, default=None)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    phash: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True, default=None)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(), server_default=func.now(), nullable=False, init=False
    )
    processing: Mapped["Processing | None"] = relationship(
        back_populates="image",
        uselist=False,
        cascade="all, delete-orphan",
        default=None,
        init=False,
    )
    annotation: Mapped["Annotation | None"] = relationship(
        back_populates="image",
        uselist=False,
        cascade="all, delete-orphan",
        default=None,
        init=False,
    )
    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="image", cascade="all, delete-orphan", default_factory=list, init=False
    )
    ingest_url: Mapped["IngestURL | None"] = relationship(
        back_populates="image", uselist=False, default=None, init=False
    )


class Processing(Base):
    __tablename__ = "processing"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    ocr_status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    ocr_model: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    ocr_updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(), nullable=True, default=None
    )
    caption_status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    caption_model: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    caption_updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(), nullable=True, default=None
    )
    embed_status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    embed_model: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    embed_dim: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    embed_updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(), nullable=True, default=None
    )
    embed_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    image: Mapped[Image] = relationship(back_populates="processing", init=False)


class Annotation(Base):
    __tablename__ = "annotations"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    caption_text: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    tags: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True, default=None)
    image: Mapped[Image] = relationship(back_populates="annotation", init=False)


class Artifact(Base):
    __tablename__ = "artifacts"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    kind: Mapped[str] = mapped_column(String(20), primary_key=True, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(), server_default=func.now(), nullable=False, init=False
    )
    image: Mapped[Image] = relationship(back_populates="artifacts", init=False)


class IndexBuild(Base):
    __tablename__ = "index_builds"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    embed_model: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    index_type: Mapped[str | None] = mapped_column(String(20), nullable=True, default=None)
    num_vectors: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    dimension: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(), server_default=func.now(), nullable=False, init=False
    )


class IngestionSource(Base):
    __tablename__ = "ingestion_sources"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    source_type: Mapped[SourceType] = mapped_column(SAEnum(SourceType), nullable=False)
    adapter_key: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    adapter_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    default_tags: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True, default=None)
    max_items_per_run: Mapped[int] = mapped_column(Integer, default=50, nullable=False)
    dataset: Mapped[str | None] = mapped_column(String(256), nullable=True, default=None)
    schedule_cron: Mapped[str | None] = mapped_column(String(100), nullable=True, default=None)
    schedule_timezone: Mapped[str] = mapped_column(String(50), default="UTC", nullable=False)
    temporal_schedule_id: Mapped[str | None] = mapped_column(
        String(256), nullable=True, default=None
    )
    last_successful_run_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, init=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        init=False,
    )
    runs: Mapped[list["SourceRun"]] = relationship(
        back_populates="source", default_factory=list, init=False
    )
    items: Mapped[list["SourceItem"]] = relationship(
        back_populates="source", default_factory=list, init=False
    )


class SourceRun(Base):
    __tablename__ = "source_runs"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("ingestion_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    trigger_mode: Mapped[SourceRunTrigger] = mapped_column(SAEnum(SourceRunTrigger), nullable=False)
    status: Mapped[SourceRunStatus] = mapped_column(
        SAEnum(SourceRunStatus), default=SourceRunStatus.RUNNING, nullable=False
    )
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True, default=None)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, init=False
    )
    source: Mapped[IngestionSource] = relationship(back_populates="runs", init=False)


class SourceItem(Base):
    __tablename__ = "source_items"
    __table_args__ = (
        UniqueConstraint("source_id", "external_item_id", name="uq_source_items_source_external"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("ingestion_sources.id", ondelete="CASCADE"), nullable=False
    )
    external_item_id: Mapped[str] = mapped_column(String(512), nullable=False)
    canonical_media_id: Mapped[str] = mapped_column(String(512), nullable=False)
    last_source_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_runs.id", ondelete="SET NULL"), nullable=True, index=True, default=None
    )
    canonical_item_url: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    title: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    published_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    raw_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True, default=None)
    first_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, init=False
    )
    last_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, init=False
    )
    source: Mapped[IngestionSource] = relationship(back_populates="items", init=False)
