"""drop unique on images.dataset

Revision ID: 03ca7ba4a3b2
Revises: 7b620cf182c8
Create Date: 2026-02-05 21:15:41.960734

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '03ca7ba4a3b2'
down_revision: str | Sequence[str] | None = '7b620cf182c8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop the unique constraint on images.dataset if it exists.
    op.execute("ALTER TABLE images DROP CONSTRAINT IF EXISTS images_dataset_key")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_unique_constraint('images_dataset_key', 'images', ['dataset'])
