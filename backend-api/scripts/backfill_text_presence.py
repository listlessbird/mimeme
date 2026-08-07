#!/usr/bin/env python3
"""Record whether each completed embedding has a text vector.

``processing.embed_text_present`` is written at embed time from the embedding
result. Rows written before that column existed are null, and the rebuild treats
null as absent. This pass lists the embedding prefix once — one Class A list
operation per 1000 keys, rather than one Class B probe per image — and settles
every null row for the model.

The default mode is read-only. Database changes require both ``--apply`` and
``--confirm APPLY``.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from sqlalchemy import func, select

from mimeme import inference, storage
from mimeme.config import ArtifactConfig, Settings
from mimeme.db import Db
from mimeme.db.schema import Processing, ProcessingStatus
from mimeme.index import ops


def _storage_config(config: ArtifactConfig) -> storage.Config:
    return storage.Config(
        endpoint_url=config.s3_endpoint_url,
        region=config.s3_region,
        access_key=config.s3_access_key_id,
        secret_key=config.s3_secret_access_key,
        bucket=config.s3_bucket,
        force_path_style=config.s3_force_path_style,
    )


async def inspect(db: Db, *, model: str) -> dict[str, object]:
    completed = (
        Processing.embed_status == ProcessingStatus.DONE,
        Processing.embed_model == model,
    )
    async with db.read_session() as session:
        total = await session.scalar(select(func.count()).select_from(Processing).where(*completed))
        unresolved = await session.scalar(
            select(func.count())
            .select_from(Processing)
            .where(*completed, Processing.embed_text_present.is_(None))
        )
        present = await session.scalar(
            select(func.count())
            .select_from(Processing)
            .where(*completed, Processing.embed_text_present.is_(True))
        )
    return {
        "model": model,
        "prefix": inference.embedding_prefix(model),
        "completed_embeddings": total,
        "text_present": present,
        "unresolved": unresolved,
        "action_required": None if unresolved == 0 else "run with --apply --confirm APPLY",
    }


async def main(args: argparse.Namespace) -> int:
    settings = Settings()
    model = args.model or settings.inference.embed_model
    db = Db(settings.database)
    artifacts = await storage.S3.open(_storage_config(settings.artifacts))
    try:
        print(json.dumps(await inspect(db, model=model), indent=2, default=str))

        if not args.apply:
            return 0
        if args.confirm != "APPLY":
            print("refusing to change the database without --confirm APPLY")
            return 2

        result = await ops.backfill_text_presence(db, artifacts, model=model)
        print(json.dumps(result.model_dump(), indent=2))
        print(json.dumps(await inspect(db, model=model), indent=2, default=str))
        return 0
    finally:
        await artifacts.close()
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="embedding model; defaults to INFERENCE_EMBED_MODEL")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
