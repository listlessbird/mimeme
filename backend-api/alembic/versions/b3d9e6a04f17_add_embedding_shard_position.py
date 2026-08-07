"""add embedding shard position

Revision ID: b3d9e6a04f17
Revises: a1c7f4d2b8e0
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b3d9e6a04f17"
down_revision: str | None = "a1c7f4d2b8e0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("processing", sa.Column("embed_shard", sa.Integer(), nullable=True))
    op.add_column("processing", sa.Column("embed_row", sa.Integer(), nullable=True))
    op.create_index("ix_processing_embed_shard", "processing", ["embed_shard"])


def downgrade() -> None:
    op.drop_index("ix_processing_embed_shard", table_name="processing")
    op.drop_column("processing", "embed_row")
    op.drop_column("processing", "embed_shard")
