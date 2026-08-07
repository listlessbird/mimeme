"""add embedding shard registry

Revision ID: c7e2a51f9b04
Revises: b3d9e6a04f17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7e2a51f9b04"
down_revision: str | None = "b3d9e6a04f17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "embedding_shards",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("embed_model", sa.Text(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("row_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sealed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("embed_model", "number", name="uq_embedding_shards_model_number"),
        sa.CheckConstraint("row_count >= 0", name="ck_embedding_shards_rows_nonneg"),
        sa.CheckConstraint("seq >= 0", name="ck_embedding_shards_seq_nonneg"),
        sa.CheckConstraint("number >= 0", name="ck_embedding_shards_number_nonneg"),
    )
    op.create_index(
        "uq_embedding_shards_one_open",
        "embedding_shards",
        ["embed_model"],
        unique=True,
        postgresql_where=sa.text("sealed = false"),
    )


def downgrade() -> None:
    op.drop_index("uq_embedding_shards_one_open", table_name="embedding_shards")
    op.drop_table("embedding_shards")
