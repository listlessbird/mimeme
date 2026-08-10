from __future__ import annotations

import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mimeme.db.schema.base import Base
from mimeme.db.schema.image import ProcessingStatus

if TYPE_CHECKING:
    from mimeme.db.schema.image import Image
    from mimeme.db.schema.job import Job
    from mimeme.db.schema.source import SourceMedia


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
    similar_image_id: Mapped[int | None] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL"), nullable=True, index=True
    )
    phash_distance: Mapped[int | None] = mapped_column(nullable=True)

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
    source_media_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_media.id", ondelete="SET NULL"), nullable=True, index=True
    )

    source_media: Mapped[SourceMedia | None] = relationship()
