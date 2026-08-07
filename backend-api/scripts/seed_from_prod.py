#!/usr/bin/env python3
"""Copy a working subset of production into the local stack.

Production is touched read-only: the database session is opened with
``default_transaction_read_only``, and object storage sees only LIST and GET.
Everything destructive happens against the local target, which must be a
loopback Postgres and a loopback S3 endpoint — the script refuses to run
otherwise.

The subset is a deterministic sample keyed on ``images.sha256``, stratified
across datasets in proportion to their production share, so the same ``--limit``
and ``--seed`` always select the same images. Rows keep their production ids so
foreign keys survive the copy; ``ingest_urls`` pointing at images outside the
subset have those references nulled.

Small provenance tables (``jobs``, ``ingestion_sources``, ``source_runs``,
``source_items``, ``ingest_urls``) are copied whole. Index artifacts are not
copied: ``search_index_state`` is seeded dirty so a local rebuild produces its
own index from the per-image embedding objects that came along with the sample.

The default mode is a dry run. Writing requires ``--apply``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import asyncpg
from aiobotocore.session import get_session
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[2]
TERRAFORM_DIR = REPO_ROOT / "terraform" / "infra"

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "minio", "postgres"}

IMAGE_TABLES = ("images", "processing", "annotations", "artifacts")
WHOLE_TABLES = ("jobs", "ingestion_sources", "source_runs", "source_items")
TRUNCATE_ORDER = (
    "ingest_urls",
    "source_items",
    "source_runs",
    "ingestion_sources",
    "artifacts",
    "annotations",
    "processing",
    "images",
    "embedding_shards",
    "index_builds",
    "jobs",
)
SERIAL_TABLES = (
    "images",
    "ingest_urls",
    "ingestion_sources",
    "source_runs",
    "source_items",
    "embedding_shards",
    "index_builds",
)


@dataclass(frozen=True)
class Bucket:
    endpoint_url: str
    region: str
    access_key: str
    secret_key: str
    bucket: str


@dataclass(frozen=True)
class Prod:
    db_url: str
    media: Bucket
    artifacts: Bucket


@dataclass(frozen=True)
class Local:
    db_url: str
    media: Bucket
    artifacts: Bucket


def _tf(name: str) -> str:
    result = subprocess.run(
        ["terraform", f"-chdir={TERRAFORM_DIR}", "output", "-raw", name],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _prod() -> Prod:
    endpoint = _tf("media_s3_endpoint_url")
    return Prod(
        db_url=_tf("db_url"),
        media=Bucket(
            endpoint_url=endpoint,
            region="auto",
            access_key=_tf("media_s3_access_key_id"),
            secret_key=_tf("media_s3_secret_access_key"),
            bucket=_tf("media_s3_bucket_name"),
        ),
        artifacts=Bucket(
            endpoint_url=_tf("artifact_s3_endpoint_url"),
            region="auto",
            access_key=_tf("artifact_s3_access_key_id"),
            secret_key=_tf("artifact_s3_secret_access_key"),
            bucket=_tf("artifact_s3_bucket_name"),
        ),
    )


def _local(args: argparse.Namespace) -> Local:
    creds = dict(region="us-east-1", access_key=args.s3_access_key, secret_key=args.s3_secret_key)
    return Local(
        db_url=args.db_url,
        media=Bucket(endpoint_url=args.s3_endpoint_url, bucket=args.media_bucket, **creds),
        artifacts=Bucket(endpoint_url=args.s3_endpoint_url, bucket=args.artifact_bucket, **creds),
    )


def _guard(target: Local) -> None:
    db_host = urlsplit(target.db_url).hostname or ""
    if db_host not in LOCAL_HOSTS:
        raise SystemExit(f"refusing to write to a non-local database host: {db_host!r}")
    for bucket in (target.media, target.artifacts):
        host = urlsplit(bucket.endpoint_url).hostname or ""
        if host not in LOCAL_HOSTS:
            raise SystemExit(f"refusing to write to a non-local S3 endpoint: {host!r}")


async def _read_only(url: str) -> asyncpg.Connection:
    return await asyncpg.connect(
        url.split("?")[0],
        statement_cache_size=0,
        server_settings={"default_transaction_read_only": "on"},
    )


def _quota(shares: list[tuple[str | None, int]], total: int, limit: int) -> dict[str | None, int]:
    if limit >= total:
        return {dataset: count for dataset, count in shares}
    quota = {dataset: min(count, limit * count // total) for dataset, count in shares}
    remaining = limit - sum(quota.values())
    for dataset, count in sorted(shares, key=lambda item: -item[1]):
        if remaining <= 0:
            break
        room = count - quota[dataset]
        take = min(room, remaining)
        quota[dataset] += take
        remaining -= take
    return quota


async def _select(prod: asyncpg.Connection, *, limit: int, seed: str) -> list[int]:
    shares = [
        (row["dataset"], row["count"])
        for row in await prod.fetch(
            "select i.dataset, count(*) as count from images i"
            " join processing p on p.image_id = i.id"
            " where p.embed_status = 'DONE' and p.embed_s3_key is not null"
            " group by 1 order by 2 desc"
        )
    ]
    total = sum(count for _, count in shares)
    if total == 0:
        raise SystemExit("production has no embedded images to sample")

    selected: list[int] = []
    for dataset, take in _quota(shares, total, limit).items():
        if take <= 0:
            continue
        rows = await prod.fetch(
            "select i.id from images i join processing p on p.image_id = i.id"
            " where p.embed_status = 'DONE' and p.embed_s3_key is not null"
            "   and i.dataset is not distinct from $1"
            " order by md5($2 || i.sha256) limit $3",
            dataset,
            seed,
            take,
        )
        selected.extend(row["id"] for row in rows)
    return sorted(selected)


async def _columns(conn: asyncpg.Connection, table: str) -> list[str]:
    rows = await conn.fetch(
        "select column_name from information_schema.columns"
        " where table_schema = 'public' and table_name = $1 order by ordinal_position",
        table,
    )
    return [row["column_name"] for row in rows]


async def _fetch(prod: asyncpg.Connection, table: str, image_ids: list[int] | None) -> list[Any]:
    columns = ", ".join(f'"{name}"' for name in await _columns(prod, table))
    if image_ids is None:
        return await prod.fetch(f"select {columns} from {table}")
    key = "id" if table == "images" else "image_id"
    return await prod.fetch(
        f"select {columns} from {table} where {key} = any($1::int[])", image_ids
    )


async def _insert(local: asyncpg.Connection, table: str, rows: list[Any]) -> int:
    if not rows:
        return 0
    columns = list(rows[0].keys())
    await local.copy_records_to_table(
        table, records=[tuple(row.values()) for row in rows], columns=columns
    )
    return len(rows)


async def _copy_rows(
    prod: asyncpg.Connection, local: asyncpg.Connection, image_ids: list[int]
) -> dict[str, int]:
    copied: dict[str, int] = {}
    for table in WHOLE_TABLES:
        copied[table] = await _insert(local, table, await _fetch(prod, table, None))
    for table in IMAGE_TABLES:
        copied[table] = await _insert(local, table, await _fetch(prod, table, image_ids))

    kept = set(image_ids)
    ingest_urls = []
    for row in await _fetch(prod, "ingest_urls", None):
        record = dict(row)
        for column in ("image_id", "duplicate_of_image_id"):
            if record[column] is not None and record[column] not in kept:
                record[column] = None
        ingest_urls.append(record)
    if ingest_urls:
        columns = list(ingest_urls[0].keys())
        await local.copy_records_to_table(
            "ingest_urls",
            records=[tuple(record[name] for name in columns) for record in ingest_urls],
            columns=columns,
        )
    copied["ingest_urls"] = len(ingest_urls)
    return copied


async def _reset_sequences(local: asyncpg.Connection) -> None:
    for table in SERIAL_TABLES:
        await local.execute(
            "select setval(pg_get_serial_sequence($1, 'id'),"
            f" coalesce((select max(id) from {table}), 1),"
            f" (select count(*) from {table}) > 0)",
            table,
        )


async def _reset_index_state(local: asyncpg.Connection) -> None:
    await local.execute("delete from search_index_state")
    await local.execute(
        "insert into search_index_state"
        " (id, desired_generation, active_generation, last_dirty_at, last_dirty_reason)"
        " values (1, 1, 0, now(), 'seeded_from_prod')"
    )


def _client(session: Any, bucket: Bucket) -> Any:
    return session.create_client(
        "s3",
        endpoint_url=bucket.endpoint_url,
        aws_access_key_id=bucket.access_key,
        aws_secret_access_key=bucket.secret_key,
        region_name=bucket.region,
    )


async def _exists(client: Any, bucket: str, key: str) -> bool:
    try:
        await client.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return False
    return True


async def _copy_object(source: Any, target: Any, *, src: Bucket, dst: Bucket, key: str) -> str:
    if await _exists(target, dst.bucket, key):
        return "skipped"
    try:
        response = await source.get_object(Bucket=src.bucket, Key=key)
    except ClientError:
        return "missing"
    async with response["Body"] as stream:
        body = await stream.read()
    await target.put_object(
        Bucket=dst.bucket,
        Key=key,
        Body=body,
        ContentType=response.get("ContentType", "binary/octet-stream"),
    )
    return "copied"


async def _copy_objects(
    prod: Prod, target: Local, plan: list[tuple[str, str]], *, concurrency: int
) -> dict[str, int]:
    tally = {"copied": 0, "skipped": 0, "missing": 0}
    session = get_session()
    async with (
        _client(session, prod.media) as prod_media,
        _client(session, prod.artifacts) as prod_artifacts,
        _client(session, target.media) as local_media,
        _client(session, target.artifacts) as local_artifacts,
    ):
        routes = {
            "media": (prod_media, local_media, prod.media, target.media),
            "artifacts": (prod_artifacts, local_artifacts, prod.artifacts, target.artifacts),
        }
        gate = asyncio.Semaphore(concurrency)

        async def run(kind: str, key: str) -> str:
            source, sink, src, dst = routes[kind]
            async with gate:
                return await _copy_object(source, sink, src=src, dst=dst, key=key)

        done = 0
        for start in range(0, len(plan), 500):
            batch = plan[start : start + 500]
            for outcome in await asyncio.gather(*(run(kind, key) for kind, key in batch)):
                tally[outcome] += 1
            done += len(batch)
            print(f"  objects {done}/{len(plan)} {tally}", flush=True)
    return tally


async def _object_plan(prod: asyncpg.Connection, image_ids: list[int]) -> list[tuple[str, str]]:
    rows = await prod.fetch(
        "select i.s3_key, p.embed_s3_key, p.embed_text_present from images i"
        " join processing p on p.image_id = i.id"
        " where i.id = any($1::int[])",
        image_ids,
    )
    plan: list[tuple[str, str]] = []
    for row in rows:
        if row["s3_key"]:
            plan.append(("media", row["s3_key"]))
        embed_key = row["embed_s3_key"]
        if embed_key:
            plan.append(("artifacts", embed_key))
            if row["embed_text_present"]:
                plan.append(("artifacts", f"{embed_key[: -len('.npy')]}_text.npy"))
    return plan


async def main(args: argparse.Namespace) -> int:
    target = _local(args)
    _guard(target)
    prod = _prod()

    source = await _read_only(prod.db_url)
    try:
        image_ids = await _select(source, limit=args.limit, seed=args.seed)
        plan = await _object_plan(source, image_ids)
        datasets = await source.fetch(
            "select coalesce(dataset, '(none)') as dataset, count(*) as count from images"
            " where id = any($1::int[]) group by 1 order by 2 desc",
            image_ids,
        )
        bytes_estimate = await source.fetchval(
            "select coalesce(sum(file_size), 0) from images where id = any($1::int[])", image_ids
        )
        print(
            json.dumps(
                {
                    "prod_db_host": urlsplit(prod.db_url).hostname,
                    "prod_media_bucket": prod.media.bucket,
                    "prod_artifact_bucket": prod.artifacts.bucket,
                    "local_db": urlsplit(target.db_url)._replace(netloc="localhost").geturl(),
                    "local_s3": target.media.endpoint_url,
                    "images": len(image_ids),
                    "by_dataset": {row["dataset"]: row["count"] for row in datasets},
                    "objects": len(plan),
                    "media_bytes": int(bytes_estimate),
                    "action_required": None if args.apply else "re-run with --apply",
                },
                indent=2,
            ),
            flush=True,
        )
        if not args.apply:
            return 0

        local = await asyncpg.connect(target.db_url.split("?")[0], statement_cache_size=0)
        try:
            async with local.transaction():
                await local.execute(
                    "update search_index_state set rebuild_job_id = null,"
                    " rebuild_target_generation = null, rebuild_claimed_at = null"
                )
                await local.execute(
                    f"truncate {', '.join(TRUNCATE_ORDER)} restart identity cascade"
                )
                copied = await _copy_rows(source, local, image_ids)
                await _reset_sequences(local)
                await _reset_index_state(local)
            print(json.dumps({"rows": copied}, indent=2), flush=True)
        finally:
            await local.close()

        tally = await _copy_objects(prod, target, plan, concurrency=args.concurrency)
        print(json.dumps({"objects": tally}, indent=2))
        return 0
    finally:
        await source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="images to sample from production")
    parser.add_argument("--seed", default="mimeme", help="sample seed; same seed, same images")
    parser.add_argument("--db-url", default="postgresql://postgres:postgres@localhost:5432/mimeme")
    parser.add_argument("--s3-endpoint-url", default="http://localhost:9000")
    parser.add_argument("--s3-access-key", default="minioadmin")
    parser.add_argument("--s3-secret-key", default="minioadmin")
    parser.add_argument("--media-bucket", default="mimeme-media")
    parser.add_argument("--artifact-bucket", default="mimeme-artifacts")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--apply", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
