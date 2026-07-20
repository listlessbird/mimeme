"""add search index state

Revision ID: 6e35c3b16410
Revises: 9b4d1f27a6c0
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "6e35c3b16410"
down_revision: str | None = "9b4d1f27a6c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "search_index_state",
        sa.Column("id", sa.SmallInteger(), autoincrement=False, nullable=False),
        sa.Column("desired_generation", sa.BigInteger(), nullable=False),
        sa.Column("active_generation", sa.BigInteger(), nullable=False),
        sa.Column("rebuild_job_id", sa.String(length=64), nullable=True),
        sa.Column("rebuild_target_generation", sa.BigInteger(), nullable=True),
        sa.Column("rebuild_claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_dirty_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_dirty_reason", sa.String(length=50), nullable=True),
        sa.Column("last_reconciled_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["rebuild_job_id"],
            ["jobs.id"],
            name="fk_search_index_state_rebuild_job_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("id = 1", name="ck_search_index_state_singleton"),
        sa.CheckConstraint("desired_generation >= 0", name="ck_search_index_state_desired_nonneg"),
        sa.CheckConstraint("active_generation >= 0", name="ck_search_index_state_active_nonneg"),
        sa.CheckConstraint(
            "active_generation <= desired_generation",
            name="ck_search_index_state_active_le_desired",
        ),
        sa.CheckConstraint(
            "(rebuild_job_id IS NULL AND rebuild_target_generation IS NULL "
            "AND rebuild_claimed_at IS NULL) OR "
            "(rebuild_job_id IS NOT NULL AND rebuild_target_generation IS NOT NULL "
            "AND rebuild_claimed_at IS NOT NULL)",
            name="ck_search_index_state_claim_all_or_none",
        ),
        sa.CheckConstraint(
            "rebuild_target_generation IS NULL OR rebuild_target_generation <= desired_generation",
            name="ck_search_index_state_target_le_desired",
        ),
    )

    op.execute(
        sa.text(
            "INSERT INTO search_index_state "
            "(id, desired_generation, active_generation, last_dirty_reason, last_dirty_at) "
            "VALUES (1, 1, 0, 'migration_baseline', now()) "
            "ON CONFLICT (id) DO NOTHING"
        )
    )


def downgrade() -> None:
    op.drop_table("search_index_state")
