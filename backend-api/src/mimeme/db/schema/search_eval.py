from __future__ import annotations

import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from mimeme.db.schema.base import Base


class SearchEvalQuery(Base):
    __tablename__ = "search_eval_queries"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'disabled')", name="ck_search_eval_queries_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    intent: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="human")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class SearchEvalPoolCandidate(Base):
    __tablename__ = "search_eval_pool_candidates"

    query_id: Mapped[int] = mapped_column(
        ForeignKey("search_eval_queries.id", ondelete="CASCADE"), primary_key=True
    )
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    image_rank: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    hybrid_rank: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    manual: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SearchEvalPoolBatch(Base):
    __tablename__ = "search_eval_pool_batches"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    index_version: Mapped[str] = mapped_column(String(50), nullable=False)
    recipe_definitions: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    query_ids: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SearchEvalPoolSource(Base):
    __tablename__ = "search_eval_pool_sources"
    __table_args__ = (
        ForeignKeyConstraint(
            ["query_id", "image_id"],
            ["search_eval_pool_candidates.query_id", "search_eval_pool_candidates.image_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("rank >= 1", name="ck_search_eval_pool_sources_rank"),
    )

    batch_id: Mapped[str] = mapped_column(
        ForeignKey("search_eval_pool_batches.id", ondelete="CASCADE"), primary_key=True
    )
    query_id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(primary_key=True)
    recipe_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    rank: Mapped[int] = mapped_column(SmallInteger, nullable=False)


class SearchEvalJudgment(Base):
    __tablename__ = "search_eval_judgments"
    __table_args__ = (
        CheckConstraint("grade >= 0 AND grade <= 3", name="ck_search_eval_judgments_grade"),
        CheckConstraint("revision >= 1", name="ck_search_eval_judgments_revision"),
    )

    query_id: Mapped[int] = mapped_column(
        ForeignKey("search_eval_queries.id", ondelete="CASCADE"), primary_key=True
    )
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    grade: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class SearchEvalSnapshot(Base):
    __tablename__ = "search_eval_snapshots"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query_count: Mapped[int] = mapped_column(Integer, nullable=False)
    judgment_count: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SearchEvalRun(Base):
    __tablename__ = "search_eval_runs"
    __table_args__ = (
        CheckConstraint("mode IN ('image', 'hybrid')", name="ck_search_eval_runs_mode"),
        CheckConstraint(
            "status IN ('queued', 'running', 'needs_judgments', 'complete', 'failed', 'cancelled')",
            name="ck_search_eval_runs_status",
        ),
        CheckConstraint(
            "phase IS NULL OR phase IN ('preparing', 'searching', 'calculating_metrics', "
            "'finalizing')",
            name="ck_search_eval_runs_phase",
        ),
        CheckConstraint(
            "progress_completed >= 0 AND progress_total >= 0 "
            "AND progress_completed <= progress_total",
            name="ck_search_eval_runs_progress",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("search_eval_experiments.id", ondelete="CASCADE"), nullable=True, index=True
    )
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("search_eval_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    recipe_id: Mapped[str] = mapped_column(String(32), nullable=False)
    recipe_definition: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="queued")
    phase: Mapped[str | None] = mapped_column(String(32), nullable=True)
    workflow_id: Mapped[str | None] = mapped_column(String(256), unique=True, nullable=True)
    progress_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    index_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    release_id: Mapped[str] = mapped_column(String(100), nullable=False)
    score_snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("search_eval_snapshots.id", ondelete="RESTRICT"), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class SearchEvalExperiment(Base):
    __tablename__ = "search_eval_experiments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("search_eval_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    index_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    recipe_definitions: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SearchEvalResult(Base):
    __tablename__ = "search_eval_results"
    __table_args__ = (
        UniqueConstraint("run_id", "query_id", "image_id", name="uq_search_eval_result_image"),
        Index("ix_search_eval_results_run_query", "run_id", "query_id"),
        CheckConstraint("rank >= 1 AND rank <= 10", name="ck_search_eval_results_rank"),
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey("search_eval_runs.id", ondelete="CASCADE"), primary_key=True
    )
    query_id: Mapped[int] = mapped_column(
        ForeignKey("search_eval_queries.id", ondelete="RESTRICT"), primary_key=True
    )
    rank: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="RESTRICT"), nullable=False
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)


class SearchEvalQueryExecution(Base):
    __tablename__ = "search_eval_query_executions"
    __table_args__ = (
        CheckConstraint("result_count >= 0", name="ck_search_eval_query_executions_count"),
        CheckConstraint("search_time_ms >= 0", name="ck_search_eval_query_executions_latency"),
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey("search_eval_runs.id", ondelete="CASCADE"), primary_key=True
    )
    query_id: Mapped[int] = mapped_column(
        ForeignKey("search_eval_queries.id", ondelete="RESTRICT"), primary_key=True
    )
    result_count: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    search_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SearchEvalScore(Base):
    __tablename__ = "search_eval_scores"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("search_eval_runs.id", ondelete="CASCADE"), primary_key=True
    )
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("search_eval_snapshots.id", ondelete="RESTRICT"), primary_key=True
    )
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SearchEvalBaseline(Base):
    __tablename__ = "search_eval_baseline"
    __table_args__ = (CheckConstraint("id = 1", name="ck_search_eval_baseline_singleton"),)

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=False)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("search_eval_runs.id", ondelete="RESTRICT"), unique=True, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
