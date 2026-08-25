"""seed Imgflip sources

Revision ID: 4a21c8e3f6b7
Revises: 8c19d4a2f6e1
Create Date: 2026-08-25
"""

from collections.abc import Sequence

from alembic import op

revision: str = "4a21c8e3f6b7"
down_revision: str | Sequence[str] | None = "8c19d4a2f6e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

def upgrade() -> None:
    # Keep the JSONB literals explicit so Alembic can also render this migration
    # in offline (--sql) mode.
    op.execute(
        """
        INSERT INTO ingestion_sources (
            name,
            source_type,
            adapter_key,
            adapter_config,
            dataset,
            schedule_cron,
            schedule_timezone,
            max_items_per_run,
            enabled
        ) VALUES
            (
                'Imgflip top 30 days',
                'API',
                'flip',
                jsonb_build_object(
                    'mode', 'top-30-days',
                    'max_templates_per_run', 15,
                    'max_meme_pages', 1
                ),
                'imgflip',
                '0 * * * *',
                'UTC',
                NULL,
                true
            ),
            (
                'Imgflip top all time',
                'API',
                'flip',
                jsonb_build_object(
                    'mode', 'top-all-time',
                    'max_templates_per_run', 15,
                    'max_meme_pages', 1
                ),
                'imgflip',
                '15 * * * *',
                'UTC',
                NULL,
                true
            ),
            (
                'Imgflip top new',
                'API',
                'flip',
                jsonb_build_object(
                    'mode', 'top-new',
                    'max_templates_per_run', 10,
                    'max_meme_pages', 1
                ),
                'imgflip',
                '45 * * * *',
                'UTC',
                NULL,
                true
            )
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DELETE FROM ingestion_sources
        WHERE adapter_key = 'flip'
          AND name IN (
              'Imgflip top 30 days',
              'Imgflip top all time',
              'Imgflip top new'
          )
        """
    )
