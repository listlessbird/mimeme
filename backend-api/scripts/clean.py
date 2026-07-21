#!/usr/bin/env python
"""Wipe all R2/S3 objects and database rows for a fresh start."""
from __future__ import annotations

import sys
from pathlib import Path

# allow `python scripts/clean.py` from the backend-api directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text

from mimeme.db.schema import Base
from mimeme.shared.db import get_engine
from mimeme.shared.services.storage import StorageService, get_s3_config


def purge_bucket(storage: StorageService) -> int:
    """Delete every object in the configured S3 bucket. Returns count deleted."""
    deleted = 0
    client = storage.client
    bucket = storage.bucket
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if not objects:
            continue
        # S3 delete_objects accepts up to 1000 keys per call
        for i in range(0, len(objects), 1000):
            batch = objects[i : i + 1000]
            client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
            deleted += len(batch)
            print(f"  deleted {deleted} objects …")

    return deleted


def truncate_tables(engine) -> list[str]:
    """TRUNCATE all application tables (CASCADE). Returns table names cleared."""
    table_names = [t.name for t in reversed(Base.metadata.sorted_tables)]
    if not table_names:
        return []

    stmt = "TRUNCATE {} CASCADE".format(", ".join(table_names))
    with engine.begin() as conn:
        conn.execute(text(stmt))

    return table_names


def main() -> None:
    config = get_s3_config()
    print(f"Bucket  : {config.bucket}")
    print(f"Endpoint: {config.endpoint_url}")

    engine = get_engine()
    db_url = str(engine.url)
    # mask password
    masked = db_url
    if "@" in db_url:
        pre, post = db_url.split("@", 1)
        if ":" in pre:
            scheme_user = pre.rsplit(":", 1)[0]
            masked = f"{scheme_user}:***@{post}"
    print(f"Database: {masked}")
    print()

    answer = input("⚠ This will DELETE all data. Continue? [y/N] ")
    if answer.strip().lower() != "y":
        print("Aborted.")
        sys.exit(0)

    print()

    # --- R2 / S3 ---
    print("Purging bucket …")
    storage = StorageService()
    n = purge_bucket(storage)
    print(f"  ✓ {n} objects deleted\n")

    # --- Database ---
    print("Truncating tables …")
    tables = truncate_tables(engine)
    for t in tables:
        print(f"  ✓ {t}")
    print(f"\n  {len(tables)} tables truncated")

    print("\nDone.")


if __name__ == "__main__":
    main()
