from __future__ import annotations

import datetime
from enum import Enum

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
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


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    type: Mapped[JobType] = mapped_column(SAEnum(JobType), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus), default=JobStatus.PENDING, nullable=False
    )
    progress: Mapped[float] = mapped_column(Text, nullable=False)
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

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_id: Mapped[int | None] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL"), nullable=True
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[Job] = relationship(back_populates="ingest_urls")
    image: Mapped[Image | None] = relationship(back_populates="ingest_url")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)

    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    dataset: Mapped[str | None] = mapped_column(String(64), nullable=False, index=True)
    original_filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    s3_etag: Mapped[str | None] = mapped_column(String(64), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    format: Mapped[str | None] = mapped_column(String(10), nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    phash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
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

    ingest_url: Mapped[IngestURL | None] = relationship(back_populates="image", uselist=False)


class Processing(Base):
    __tablename__ = "processing"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )

    ocr_status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    ocr_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ocr_updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    caption_status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    caption_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    caption_updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    embed_status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    embed_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embed_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embed_updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
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

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    model_version: Mapped[str] = mapped_column(String(255), nullable=False)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    image: Mapped[Image] = relationship(back_populates="artifacts")

    __table_args__ = (
        UniqueConstraint(
            "image_id",
            "kind",
            "model_version",
            name="uq_artifacts_image_kind_version",
        ),
    )


class IndexBuild(Base):
    __tablename__ = "index_builds"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    embed_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    index_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    num_vectors: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=False, nullable=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
