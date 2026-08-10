"""add source media and known facts

Revision ID: f4a8c2d91e70
Revises: c7e2a51f9b04
Create Date: 2026-08-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f4a8c2d91e70"
down_revision: str | Sequence[str] | None = "c7e2a51f9b04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FK_INGEST_URLS_SOURCE_MEDIA = "fk_ingest_urls_source_media_id_source_media"
FK_INGEST_URLS_SIMILAR_IMAGE = "fk_ingest_urls_similar_image_id_images"


def upgrade() -> None:
    op.add_column("source_runs", sa.Column("discovery_key", sa.String(length=128), nullable=True))
    op.add_column(
        "source_runs",
        sa.Column("discovered_count", sa.Integer(), server_default="0", nullable=False),
    )
    op.execute(
        """
        UPDATE source_runs AS run
        SET discovered_count = counts.value
        FROM (
            SELECT last_source_run_id, count(*) AS value
            FROM source_items
            WHERE last_source_run_id IS NOT NULL
            GROUP BY last_source_run_id
        ) AS counts
        WHERE run.id = counts.last_source_run_id
        """
    )
    op.create_index(
        "uq_source_runs_discovery_key",
        "source_runs",
        ["discovery_key"],
        unique=True,
        postgresql_where=sa.text("discovery_key IS NOT NULL"),
    )
    op.add_column("ingest_urls", sa.Column("similar_image_id", sa.Integer(), nullable=True))
    op.add_column("ingest_urls", sa.Column("phash_distance", sa.Integer(), nullable=True))
    op.create_index("ix_ingest_urls_similar_image_id", "ingest_urls", ["similar_image_id"])
    op.create_foreign_key(
        FK_INGEST_URLS_SIMILAR_IMAGE,
        "ingest_urls",
        "images",
        ["similar_image_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.add_column(
        "annotations", sa.Column("caption_context_sha256", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "annotations", sa.Column("caption_prompt_version", sa.String(length=32), nullable=True)
    )
    op.add_column("processing", sa.Column("embed_text_sha256", sa.String(length=64), nullable=True))
    op.add_column(
        "processing", sa.Column("embed_recipe_version", sa.String(length=32), nullable=True)
    )
    op.add_column(
        "source_items",
        sa.Column(
            "known_facts",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "source_items", sa.Column("known_facts_sha256", sa.String(length=64), nullable=True)
    )
    op.create_table(
        "source_media",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_item_id", sa.Integer(), nullable=False),
        sa.Column("external_media_id", sa.Text(), nullable=False),
        sa.Column("media_url", sa.Text(), nullable=False),
        sa.Column("canonical_media_url", sa.Text(), nullable=True),
        sa.Column("raw_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_item_id"], ["source_items.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_item_id", "external_media_id", name="uq_source_media_item_external"
        ),
    )
    op.create_index("ix_source_media_source_item_id", "source_media", ["source_item_id"])

    op.execute(
        """
        INSERT INTO source_media (
            source_item_id, external_media_id, media_url, canonical_media_url,
            first_seen_at, last_seen_at
        )
        SELECT id, 'primary', canonical_image_url, canonical_image_url,
               first_seen_at, last_seen_at
        FROM source_items
        WHERE canonical_image_url IS NOT NULL
        """
    )

    op.add_column("ingest_urls", sa.Column("source_media_id", sa.Integer(), nullable=True))
    op.create_index("ix_ingest_urls_source_media_id", "ingest_urls", ["source_media_id"])
    op.create_foreign_key(
        FK_INGEST_URLS_SOURCE_MEDIA,
        "ingest_urls",
        "source_media",
        ["source_media_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.execute(
        """
        UPDATE ingest_urls AS ingest
        SET source_media_id = media.id
        FROM source_media AS media
        WHERE ingest.source_item_id = media.source_item_id
          AND media.external_media_id = 'primary'
        """
    )
    op.drop_column("source_items", "canonical_image_url")


def downgrade() -> None:
    op.add_column("source_items", sa.Column("canonical_image_url", sa.Text(), nullable=True))
    op.execute(
        """
        UPDATE source_items AS item
        SET canonical_image_url = media.canonical_media_url
        FROM source_media AS media
        WHERE media.source_item_id = item.id
          AND media.external_media_id = 'primary'
        """
    )
    op.drop_constraint(FK_INGEST_URLS_SOURCE_MEDIA, "ingest_urls", type_="foreignkey")
    op.drop_index("ix_ingest_urls_source_media_id", table_name="ingest_urls")
    op.drop_column("ingest_urls", "source_media_id")
    op.drop_index("ix_source_media_source_item_id", table_name="source_media")
    op.drop_table("source_media")
    op.drop_column("source_items", "known_facts_sha256")
    op.drop_column("source_items", "known_facts")
    op.drop_column("processing", "embed_recipe_version")
    op.drop_column("processing", "embed_text_sha256")
    op.drop_column("annotations", "caption_prompt_version")
    op.drop_column("annotations", "caption_context_sha256")
    op.drop_constraint(FK_INGEST_URLS_SIMILAR_IMAGE, "ingest_urls", type_="foreignkey")
    op.drop_index("ix_ingest_urls_similar_image_id", table_name="ingest_urls")
    op.drop_column("ingest_urls", "phash_distance")
    op.drop_column("ingest_urls", "similar_image_id")
    op.execute("DROP INDEX IF EXISTS uq_source_runs_discovery_key")
    op.execute("ALTER TABLE source_runs DROP COLUMN IF EXISTS discovered_count")
    op.execute("ALTER TABLE source_runs DROP COLUMN IF EXISTS discovery_key")
