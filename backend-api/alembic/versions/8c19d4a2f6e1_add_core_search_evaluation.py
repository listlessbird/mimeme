"""add core search evaluation

Revision ID: 8c19d4a2f6e1
Revises: f4a8c2d91e70
Create Date: 2026-08-24 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8c19d4a2f6e1"
down_revision: str | Sequence[str] | None = "f4a8c2d91e70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "search_eval_queries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('active', 'disabled')", name="ck_search_eval_queries_status"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("text"),
    )
    op.create_table(
        "search_eval_pool_candidates",
        sa.Column("query_id", sa.Integer(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("image_rank", sa.SmallInteger(), nullable=True),
        sa.Column("hybrid_rank", sa.SmallInteger(), nullable=True),
        sa.Column("manual", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["query_id"], ["search_eval_queries.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("query_id", "image_id"),
    )
    op.create_table(
        "search_eval_judgments",
        sa.Column("query_id", sa.Integer(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("grade", sa.SmallInteger(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("grade >= 0 AND grade <= 3", name="ck_search_eval_judgments_grade"),
        sa.CheckConstraint("revision >= 1", name="ck_search_eval_judgments_revision"),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["query_id"], ["search_eval_queries.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("query_id", "image_id"),
    )
    op.create_table(
        "search_eval_snapshots",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("query_hash", sa.String(length=64), nullable=False),
        sa.Column("query_count", sa.Integer(), nullable=False),
        sa.Column("judgment_count", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_hash"),
    )
    op.create_index("ix_search_eval_snapshots_query_hash", "search_eval_snapshots", ["query_hash"])
    op.create_table(
        "search_eval_runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("snapshot_id", sa.String(length=32), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("phase", sa.String(length=32), nullable=True),
        sa.Column("workflow_id", sa.String(length=256), nullable=True),
        sa.Column("progress_completed", sa.Integer(), nullable=False),
        sa.Column("progress_total", sa.Integer(), nullable=False),
        sa.Column("index_version", sa.String(length=50), nullable=True),
        sa.Column("release_id", sa.String(length=100), nullable=False),
        sa.Column("score_snapshot_id", sa.String(length=32), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("mode IN ('image', 'hybrid')", name="ck_search_eval_runs_mode"),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'needs_judgments', 'complete', 'failed', "
            "'cancelled')",
            name="ck_search_eval_runs_status",
        ),
        sa.CheckConstraint(
            "phase IS NULL OR phase IN ('preparing', 'searching', 'calculating_metrics', "
            "'finalizing')",
            name="ck_search_eval_runs_phase",
        ),
        sa.CheckConstraint(
            "progress_completed >= 0 AND progress_total >= 0 "
            "AND progress_completed <= progress_total",
            name="ck_search_eval_runs_progress",
        ),
        sa.ForeignKeyConstraint(
            ["score_snapshot_id"], ["search_eval_snapshots.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["snapshot_id"], ["search_eval_snapshots.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workflow_id"),
    )
    op.create_table(
        "search_eval_query_executions",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("query_id", sa.Integer(), nullable=False),
        sa.Column("result_count", sa.SmallInteger(), nullable=False),
        sa.Column("search_time_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "result_count >= 0", name="ck_search_eval_query_executions_count"
        ),
        sa.CheckConstraint(
            "search_time_ms >= 0", name="ck_search_eval_query_executions_latency"
        ),
        sa.ForeignKeyConstraint(["query_id"], ["search_eval_queries.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["run_id"], ["search_eval_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "query_id"),
    )
    op.create_table(
        "search_eval_results",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("query_id", sa.Integer(), nullable=False),
        sa.Column("rank", sa.SmallInteger(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.CheckConstraint("rank >= 1 AND rank <= 10", name="ck_search_eval_results_rank"),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["query_id"], ["search_eval_queries.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["run_id"], ["search_eval_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "query_id", "rank"),
        sa.UniqueConstraint("run_id", "query_id", "image_id", name="uq_search_eval_result_image"),
    )
    op.create_index(
        "ix_search_eval_results_run_query", "search_eval_results", ["run_id", "query_id"]
    )
    op.create_table(
        "search_eval_scores",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("snapshot_id", sa.String(length=32), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["search_eval_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["search_eval_snapshots.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("run_id", "snapshot_id"),
    )
    op.create_table(
        "search_eval_baseline",
        sa.Column("id", sa.SmallInteger(), autoincrement=False, nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("id = 1", name="ck_search_eval_baseline_singleton"),
        sa.ForeignKeyConstraint(["run_id"], ["search_eval_runs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )


def downgrade() -> None:
    op.drop_table("search_eval_baseline")
    op.drop_table("search_eval_scores")
    op.drop_index("ix_search_eval_results_run_query", table_name="search_eval_results")
    op.drop_table("search_eval_results")
    op.drop_table("search_eval_query_executions")
    op.drop_table("search_eval_runs")
    op.drop_index("ix_search_eval_snapshots_query_hash", table_name="search_eval_snapshots")
    op.drop_table("search_eval_snapshots")
    op.drop_table("search_eval_judgments")
    op.drop_table("search_eval_pool_candidates")
    op.drop_table("search_eval_queries")
