"""add typed image ingest inputs

Revision ID: 9b4d1f27a6c0
Revises: d0d65c1bccf2
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9b4d1f27a6c0"
down_revision: str | None = "d0d65c1bccf2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

input_kind = sa.Enum("remote_image_url", "staged_upload", name="ingestinputkind")


def upgrade() -> None:
    input_kind.create(op.get_bind(), checkfirst=True)
    op.add_column(
        "ingest_urls",
        sa.Column(
            "input_kind",
            input_kind,
            nullable=False,
            server_default="remote_image_url",
        ),
    )
    op.add_column("ingest_urls", sa.Column("artifact_key", sa.Text(), nullable=True))
    op.alter_column("ingest_urls", "url", existing_type=sa.Text(), nullable=True)
    op.create_check_constraint(
        "ck_ingest_urls_input_payload",
        "ingest_urls",
        "(input_kind = 'remote_image_url' AND url IS NOT NULL AND artifact_key IS NULL) OR "
        "(input_kind = 'staged_upload' AND url IS NULL AND artifact_key IS NOT NULL)",
    )
    op.alter_column("ingest_urls", "input_kind", server_default=None)


def downgrade() -> None:
    op.drop_constraint("ck_ingest_urls_input_payload", "ingest_urls", type_="check")
    op.execute("DELETE FROM ingest_urls WHERE input_kind = 'staged_upload'")
    op.alter_column("ingest_urls", "url", existing_type=sa.Text(), nullable=False)
    op.drop_column("ingest_urls", "artifact_key")
    op.drop_column("ingest_urls", "input_kind")
    input_kind.drop(op.get_bind(), checkfirst=True)
