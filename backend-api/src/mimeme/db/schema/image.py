from __future__ import annotations

import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mimeme.db.schema.base import Base

if TYPE_CHECKING:
    from mimeme.db.schema.ingest import IngestURL


class ProcessingStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


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
