"""add phash dedupe prevenance to ingest_urls

Revision ID: a093b27894ea
Revises: c05d42b69c83
Create Date: 2026-06-04 18:04:36.705070

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "a093b27894ea"
down_revision: str | Sequence[str] | None = "c05d42b69c83"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
