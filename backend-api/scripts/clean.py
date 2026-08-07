#!/usr/bin/env python
"""Wipe configured object stores and database rows after an explicit prompt."""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from mimeme import storage
from mimeme.config import ArtifactConfig, MediaConfig, Settings
from mimeme.db import Db
from mimeme.db.schema import Base


def _config(value: MediaConfig | ArtifactConfig) -> storage.Config:
    return storage.Config(
        endpoint_url=value.s3_endpoint_url,
        region=value.s3_region,
        access_key=value.s3_access_key_id,
        secret_key=value.s3_secret_access_key,
        bucket=value.s3_bucket,
        force_path_style=value.s3_force_path_style,
    )


async def _purge(store: storage.Store) -> int:
    objects = [item.object async for item in store.list()]
    for item in objects:
        await store.delete(item)
    return len(objects)


async def main() -> None:
    settings = Settings()
    print(f"Media bucket: {settings.media.s3_bucket}")
    print(f"Artifact bucket: {settings.artifacts.s3_bucket}")
    print(f"Database: {settings.database.url_str.split('@')[-1]}")
    answer = input("This will DELETE all configured data. Continue? [y/N] ")
    if answer.strip().lower() != "y":
        print("Aborted.")
        return

    db = Db(settings.database)
    media = await storage.S3.open(_config(settings.media))
    artifacts = await storage.S3.open(_config(settings.artifacts))
    try:
        print(f"Deleted {await _purge(media)} media objects")
        print(f"Deleted {await _purge(artifacts)} artifact objects")
        tables = [table.name for table in reversed(Base.metadata.sorted_tables)]
        if tables:
            async with db.engine.begin() as connection:
                await connection.execute(text(f"TRUNCATE {', '.join(tables)} CASCADE"))
        print(f"Truncated {len(tables)} tables")
    finally:
        await artifacts.close()
        await media.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
