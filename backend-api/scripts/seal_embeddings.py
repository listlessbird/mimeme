#!/usr/bin/env python3
"""Seal individual embedding objects into shard matrices.

A rebuild reads one object per image, twice. Sealing packs a fixed number of
vectors into one 2-D float32 matrix per family, so a rebuild reads one object
per shard instead. This script drives the same ``index.pack`` code path the
rebuild workflow's seal activity uses; it is not a separate migration
mechanism. The matrix work runs in the compute service, so a compute gateway
must be reachable at ``COMPUTE_GATEWAY_URL``.

Sealing is resumable: shard positions are recorded only after the shard objects
upload and verify, so a killed run re-plans the rows it did not finish. It is
also single-flight — a Postgres advisory lock keyed on the embedding model
refuses a second concurrent seal, including one started by a rebuild.
Individual objects are left in place.

The default mode is read-only. Sealing requires both ``--apply`` and
``--confirm APPLY``.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx

from mimeme.config import Settings
from mimeme.db import Db
from mimeme.index import pack
from mimeme.index.local import Local


def _report(target: pack.Plan) -> dict[str, object]:
    return {
        "model": target.model,
        "shard_rows": target.shard_rows,
        "unsealed": target.unsealed,
        "planned_shards": len(target.shards),
        "planned_rows": sum(len(shard.members) for shard in target.shards),
        "unsealed_tail_after": target.tail,
        "first_shard": target.shards[0].number if target.shards else None,
        "last_shard": target.shards[-1].number if target.shards else None,
        "first_shard_keys": (
            [target.shards[0].image_key, target.shards[0].text_key] if target.shards else []
        ),
        "action_required": (None if not target.shards else "run with --apply --confirm APPLY"),
    }


async def main(args: argparse.Namespace) -> int:
    settings = Settings()
    model = args.model or settings.inference.embed_model
    shard_rows = args.shard_rows or settings.index.shard_rows
    db = Db(settings.database)
    http = httpx.AsyncClient(timeout=args.call_timeout_s)
    try:
        target = await pack.plan(db, model=model, shard_rows=shard_rows, max_shards=args.max_shards)
        print(json.dumps(_report(target), indent=2))

        if not args.apply:
            return 0
        if args.confirm != "APPLY":
            print("refusing to seal without --confirm APPLY")
            return 2
        if not target.shards:
            return 0

        client = Local(
            http,
            base_url=settings.compute.gateway_url,
            poll_interval_s=settings.compute.poll_interval_s,
        )
        try:
            sealed = await pack.seal(
                db,
                client,
                job_id=args.job_id,
                model=model,
                shard_rows=shard_rows,
                max_shards=args.max_shards,
            )
        except pack.Busy as busy:
            print(json.dumps({"skipped": str(busy)}, indent=2))
            return 3
        print(json.dumps(sealed.model_dump(), indent=2))
        remaining = await pack.plan(db, model=model, shard_rows=shard_rows)
        print(json.dumps(_report(remaining), indent=2))
        return 0
    finally:
        await http.aclose()
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="embedding model; defaults to INFERENCE_EMBED_MODEL")
    parser.add_argument(
        "--shard-rows", type=int, help="rows per shard; defaults to INDEX_SHARD_ROWS"
    )
    parser.add_argument("--max-shards", type=int, help="seal at most this many shards in one run")
    parser.add_argument("--job-id", default="seal-backlog")
    parser.add_argument("--call-timeout-s", type=float, default=900.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
