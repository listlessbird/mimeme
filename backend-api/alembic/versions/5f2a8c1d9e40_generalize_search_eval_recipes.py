"""generalize search evaluation recipes

Revision ID: 5f2a8c1d9e40
Revises: 4a21c8e3f6b7
Create Date: 2026-08-29 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5f2a8c1d9e40"
down_revision: str | Sequence[str] | None = "4a21c8e3f6b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "search_eval_pool_batches",
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("index_version", sa.String(50), nullable=False),
        sa.Column("recipe_definitions", sa.JSON(), nullable=False),
        sa.Column("query_ids", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "search_eval_experiments",
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("snapshot_id", sa.String(32), nullable=False),
        sa.Column("index_version", sa.String(50), nullable=True),
        sa.Column("recipe_definitions", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["snapshot_id"], ["search_eval_snapshots.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.add_column("search_eval_runs", sa.Column("experiment_id", sa.String(32), nullable=True))
    op.add_column("search_eval_runs", sa.Column("recipe_id", sa.String(32), nullable=True))
    op.add_column("search_eval_runs", sa.Column("recipe_definition", sa.JSON(), nullable=True))
    op.create_foreign_key(
        "fk_search_eval_runs_experiment",
        "search_eval_runs",
        "search_eval_experiments",
        ["experiment_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_search_eval_runs_experiment_id", "search_eval_runs", ["experiment_id"])
    op.execute(
        """
        INSERT INTO search_eval_experiments
            (id, snapshot_id, index_version, recipe_definitions)
        SELECT id, snapshot_id, index_version,
            json_build_array(json_build_object(
                'id', CASE mode WHEN 'image' THEN 'image_only' ELSE 'image_siglip_text' END,
                'version', 1,
                'label', CASE mode WHEN 'image' THEN 'Image only' ELSE 'Image and SigLIP text' END,
                'retrievers', CASE mode
                    WHEN 'image' THEN json_build_array('siglip_image')
                    ELSE json_build_array('siglip_image', 'siglip_text') END,
                'candidate_depth', 1000,
                'rrf_k', 60
            ))
        FROM search_eval_runs
        """
    )
    op.execute(
        """
        UPDATE search_eval_runs
        SET experiment_id = id,
            recipe_id = CASE mode WHEN 'image' THEN 'image_only' ELSE 'image_siglip_text' END,
            recipe_definition = (SELECT recipe_definitions->0
                FROM search_eval_experiments WHERE id = search_eval_runs.id)
        """
    )
    op.alter_column("search_eval_runs", "recipe_id", nullable=False)
    op.alter_column("search_eval_runs", "recipe_definition", nullable=False)
    op.create_table(
        "search_eval_pool_sources",
        sa.Column("batch_id", sa.String(32), nullable=False),
        sa.Column("query_id", sa.Integer(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("recipe_id", sa.String(32), nullable=False),
        sa.Column("rank", sa.SmallInteger(), nullable=False),
        sa.CheckConstraint("rank >= 1", name="ck_search_eval_pool_sources_rank"),
        sa.ForeignKeyConstraint(["batch_id"], ["search_eval_pool_batches.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["query_id", "image_id"],
            ["search_eval_pool_candidates.query_id", "search_eval_pool_candidates.image_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("batch_id", "query_id", "image_id", "recipe_id"),
    )
    op.execute(
        """
        INSERT INTO search_eval_pool_batches
            (id, index_version, recipe_definitions, query_ids)
        SELECT md5('legacy-search-pool-' || query_id::text), 'legacy-unknown',
            json_build_array(
                json_build_object('id', 'image_only', 'version', 1,
                    'label', 'Image only', 'retrievers', json_build_array('siglip_image'),
                    'candidate_depth', 1000, 'rrf_k', 60),
                json_build_object('id', 'image_siglip_text', 'version', 1,
                    'label', 'Image and SigLIP text',
                    'retrievers', json_build_array('siglip_image', 'siglip_text'),
                    'candidate_depth', 1000, 'rrf_k', 60)
            ), json_build_array(query_id)
        FROM search_eval_pool_candidates
        WHERE image_rank IS NOT NULL OR hybrid_rank IS NOT NULL
        GROUP BY query_id
        """
    )
    op.execute(
        """
        INSERT INTO search_eval_pool_sources (batch_id, query_id, image_id, recipe_id, rank)
        SELECT md5('legacy-search-pool-' || query_id::text), query_id, image_id,
            'image_only', image_rank
        FROM search_eval_pool_candidates WHERE image_rank IS NOT NULL
        UNION ALL
        SELECT md5('legacy-search-pool-' || query_id::text), query_id, image_id,
            'image_siglip_text', hybrid_rank
        FROM search_eval_pool_candidates WHERE hybrid_rank IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE search_eval_runs
        SET mode = CASE recipe_id WHEN 'image_only' THEN 'image' ELSE 'hybrid' END
        """
    )
    op.execute(
        """
        UPDATE search_eval_pool_candidates AS candidate
        SET image_rank = source.rank
        FROM search_eval_pool_sources AS source
        WHERE source.query_id = candidate.query_id
          AND source.image_id = candidate.image_id
          AND source.recipe_id = 'image_only'
        """
    )
    op.execute(
        """
        UPDATE search_eval_pool_candidates AS candidate
        SET hybrid_rank = source.rank
        FROM search_eval_pool_sources AS source
        WHERE source.query_id = candidate.query_id
          AND source.image_id = candidate.image_id
          AND source.recipe_id = 'image_siglip_text'
        """
    )
    op.drop_table("search_eval_pool_sources")
    op.drop_index("ix_search_eval_runs_experiment_id", table_name="search_eval_runs")
    op.drop_constraint("fk_search_eval_runs_experiment", "search_eval_runs", type_="foreignkey")
    op.drop_column("search_eval_runs", "recipe_definition")
    op.drop_column("search_eval_runs", "recipe_id")
    op.drop_column("search_eval_runs", "experiment_id")
    op.drop_table("search_eval_experiments")
    op.drop_table("search_eval_pool_batches")
