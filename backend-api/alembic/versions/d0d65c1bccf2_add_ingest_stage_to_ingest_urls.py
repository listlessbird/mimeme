"""add ingest stage to ingest_urls

Revision ID: d0d65c1bccf2
Revises: b745c57e8993
Create Date: 2026-06-26 03:30:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd0d65c1bccf2'
down_revision: str | Sequence[str] | None = 'b745c57e8993'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Pipeline-position enum backing `ingest_urls.stage`. Positions only — never an
# outcome; failure is carried by `status`/`error_message`.
INGEST_STAGE = postgresql.ENUM(
    "QUEUED",
    "DOWNLOADING",
    "PROCESSING",
    "ANNOTATING",
    "EMBEDDING",
    "COMPLETE",
    "DEDUPED",
    name="ingeststage",
)


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    INGEST_STAGE.create(bind, checkfirst=True)
    # server_default backfills existing rows; dropped afterwards so the column
    # matches the ORM (python-side default only).
    op.add_column(
        "ingest_urls",
        sa.Column(
            "stage",
            INGEST_STAGE,
            nullable=False,
            server_default="QUEUED",
        ),
    )
    op.alter_column("ingest_urls", "stage", server_default=None)
    op.add_column(
        "ingest_urls",
        sa.Column("stage_updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("ingest_urls", "stage_updated_at")
    op.drop_column("ingest_urls", "stage")
    INGEST_STAGE.drop(op.get_bind(), checkfirst=True)
