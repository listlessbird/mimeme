from __future__ import annotations

import uuid

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, text

_PREVIOUS = "4a21c8e3f6b7"
_REVISION = "5f2a8c1d9e40"


def test_recipe_migration_preserves_runs_pool_candidates_and_judgments(
    db_engine: Engine,
) -> None:
    schema = f"migration_{uuid.uuid4().hex}"
    config = Config("alembic.ini")
    scripts = ScriptDirectory.from_config(config)

    with db_engine.connect() as connection, connection.begin():
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            revisions = list(scripts.walk_revisions(base="base", head=_PREVIOUS))
            for revision in reversed(revisions):
                revision.module.upgrade()

            image_id = connection.execute(
                text("INSERT INTO images (sha256) VALUES (:sha256) RETURNING id"),
                {"sha256": "a" * 64},
            ).scalar_one()
            query_id = connection.execute(
                text(
                    "INSERT INTO search_eval_queries (text, intent, source, status) "
                    "VALUES ('query', 'situation', 'human', 'active') RETURNING id"
                )
            ).scalar_one()
            connection.execute(
                text(
                    "INSERT INTO search_eval_pool_candidates "
                    "(query_id, image_id, image_rank, hybrid_rank, manual) "
                    "VALUES (:query_id, :image_id, 2, 1, false)"
                ),
                {"query_id": query_id, "image_id": image_id},
            )
            connection.execute(
                text(
                    "INSERT INTO search_eval_judgments "
                    "(query_id, image_id, grade, revision) VALUES (:query_id, :image_id, 3, 1)"
                ),
                {"query_id": query_id, "image_id": image_id},
            )
            connection.execute(
                text(
                    "INSERT INTO search_eval_snapshots "
                    "(id, content_hash, query_hash, query_count, judgment_count, payload) "
                    "VALUES ('snapshot', :content_hash, :query_hash, 1, 1, '{}'::json)"
                ),
                {"content_hash": "b" * 64, "query_hash": "c" * 64},
            )
            connection.execute(
                text(
                    "INSERT INTO search_eval_runs "
                    "(id, snapshot_id, mode, status, progress_completed, progress_total, "
                    "index_version, release_id) VALUES "
                    "('run', 'snapshot', 'hybrid', 'complete', 1, 1, 'index-v1', 'release')"
                )
            )

            migration = scripts.get_revision(_REVISION)
            assert migration is not None
            migration.module.upgrade()

            migrated = connection.execute(
                text("SELECT recipe_id FROM search_eval_runs WHERE id = 'run'")
            ).scalar_one()
            sources = connection.execute(
                text("SELECT recipe_id, rank FROM search_eval_pool_sources ORDER BY recipe_id")
            ).all()
            judgment = connection.execute(
                text("SELECT grade FROM search_eval_judgments")
            ).scalar_one()
            assert migrated == "image_siglip_text"
            assert sources == [("image_only", 2), ("image_siglip_text", 1)]
            assert judgment == 3

            migration.module.downgrade()

            legacy = connection.execute(
                text("SELECT mode FROM search_eval_runs WHERE id = 'run'")
            ).scalar_one()
            ranks = connection.execute(
                text(
                    "SELECT image_rank, hybrid_rank FROM search_eval_pool_candidates "
                    "WHERE query_id = :query_id AND image_id = :image_id"
                ),
                {"query_id": query_id, "image_id": image_id},
            ).one()
            judgment = connection.execute(
                text("SELECT grade FROM search_eval_judgments")
            ).scalar_one()
            assert legacy == "hybrid"
            assert ranks == (2, 1)
            assert judgment == 3
