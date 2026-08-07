"""add phash dedup provenance to ingest_urls

Revision ID: e71941847b1f
Revises: c05d42b69c83
Create Date: 2026-06-04 18:08:29.812297

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e71941847b1f'
down_revision: str | Sequence[str] | None = 'c05d42b69c83'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FK_DUPLICATE_OF_IMAGE = "fk_ingest_urls_duplicate_of_image_id_images"


def upgrade() -> None:
    """Upgrade schema."""
    # Postgres needs the enum type created explicitly; add_column won't emit
    # CREATE TYPE for it. create_type=False stops the column DDL from trying
    # to create it a second time.
    duplicate_reason = postgresql.ENUM(
        "SOURCE_SEEN", "SHA256", "PHASH", name="duplicatereason", create_type=False
    )
    duplicate_reason.create(op.get_bind(), checkfirst=True)

    op.add_column(
        "ingest_urls", sa.Column("duplicate_reason", duplicate_reason, nullable=True)
    )
    op.add_column(
        "ingest_urls", sa.Column("duplicate_of_image_id", sa.Integer(), nullable=True)
    )
    op.create_index(
        op.f("ix_ingest_urls_duplicate_of_image_id"),
        "ingest_urls",
        ["duplicate_of_image_id"],
        unique=False,
    )
    op.create_foreign_key(
        FK_DUPLICATE_OF_IMAGE,
        "ingest_urls",
        "images",
        ["duplicate_of_image_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(FK_DUPLICATE_OF_IMAGE, "ingest_urls", type_="foreignkey")
    op.drop_index(op.f("ix_ingest_urls_duplicate_of_image_id"), table_name="ingest_urls")
    op.drop_column("ingest_urls", "duplicate_of_image_id")
    op.drop_column("ingest_urls", "duplicate_reason")
    postgresql.ENUM(name="duplicatereason").drop(op.get_bind(), checkfirst=True)
