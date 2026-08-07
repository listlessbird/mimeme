from __future__ import annotations

import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mimeme.db.schema.base import Base


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
