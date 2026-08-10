#!/usr/bin/env -S uv run
"""Run the production KYM adapter and write a local inspection manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from urllib.parse import urlparse

import httpx

from mimeme.source.fetch import Fetcher
from mimeme.source.http import Http
from mimeme.source.kym import KymAdapter

DEFAULT_OUTPUT = Path("scripts/kym/output/confirmed-newest.json")


async def run(args: argparse.Namespace) -> int:
    config = {
        "start_page": args.start_page,
        "max_entries_per_run": args.max_entries,
        "max_photo_pages_per_entry": args.max_photo_pages,
        "max_items_per_run": args.max_images,
        "delay_seconds": args.delay,
        "timeout_seconds": args.timeout,
        "retries": args.retries,
        "impersonate": args.impersonate,
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=args.timeout) as client:
        async with Fetcher(
            Http(client),
            delay_seconds=args.delay,
            timeout_seconds=args.timeout,
            retries=args.retries,
            impersonate=args.impersonate,
        ) as fetcher:
            items = [
                item async for item in KymAdapter().discover(config, fetcher=fetcher, rng=Random())
            ]
        if args.download_images:
            await _download(client, items, args.output.parent / "images")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": "kym",
                "scraped_at": datetime.now(UTC).isoformat(),
                "items": [item.model_dump(mode="json") for item in items],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"items={len(items)} media={sum(len(item.media) for item in items)} output={args.output}")
    return 0


async def _download(client: httpx.AsyncClient, items, root: Path) -> None:  # noqa: ANN001
    semaphore = asyncio.Semaphore(4)

    async def one(slug: str, media) -> None:  # noqa: ANN001
        suffix = Path(urlparse(media.media_url).path).suffix.lower() or ".jpg"
        path = root / slug / f"{media.external_media_id.replace('/', '_')}{suffix}"
        if path.exists() and path.stat().st_size > 0:
            return
        async with semaphore:
            response = await client.get(media.media_url)
            response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)

    await asyncio.gather(
        *(one(item.external_item_id, media) for item in items for media in item.media)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-entries", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--max-photo-pages", type=int, default=100)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--impersonate", default="chrome")
    parser.add_argument("--download-images", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
