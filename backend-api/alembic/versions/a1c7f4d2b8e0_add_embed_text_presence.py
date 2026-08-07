"""add embed text presence

Revision ID: a1c7f4d2b8e0
Revises: 6e35c3b16410
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1c7f4d2b8e0"
down_revision: str | None = "6e35c3b16410"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("processing", sa.Column("embed_text_present", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("processing", "embed_text_present")
