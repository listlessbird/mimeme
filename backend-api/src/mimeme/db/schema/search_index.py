from __future__ import annotations

import datetime
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from mimeme.db.schema.base import Base


class RebuildTrigger(StrEnum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"


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


class SearchIndexState(Base):
    __tablename__ = "search_index_state"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_search_index_state_singleton"),
        CheckConstraint("desired_generation >= 0", name="ck_search_index_state_desired_nonneg"),
        CheckConstraint("active_generation >= 0", name="ck_search_index_state_active_nonneg"),
        CheckConstraint(
            "active_generation <= desired_generation",
            name="ck_search_index_state_active_le_desired",
        ),
        CheckConstraint(
            "(rebuild_job_id IS NULL AND rebuild_target_generation IS NULL "
            "AND rebuild_claimed_at IS NULL) OR "
            "(rebuild_job_id IS NOT NULL AND rebuild_target_generation IS NOT NULL "
            "AND rebuild_claimed_at IS NOT NULL)",
            name="ck_search_index_state_claim_all_or_none",
        ),
        CheckConstraint(
            "rebuild_target_generation IS NULL OR rebuild_target_generation <= desired_generation",
            name="ck_search_index_state_target_le_desired",
        ),
    )

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=False)
    desired_generation: Mapped[int] = mapped_column(BigInteger, nullable=False)
    active_generation: Mapped[int] = mapped_column(BigInteger, nullable=False)

    rebuild_job_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("jobs.id", ondelete="RESTRICT"), nullable=True
    )
    rebuild_target_generation: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    rebuild_claimed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    last_dirty_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_dirty_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    last_reconciled_at: Mapped[datetime.datetime | None] = mapped_column(
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
