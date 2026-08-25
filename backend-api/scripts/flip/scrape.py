#!/usr/bin/env -S uv run
"""Run the production Imgflip adapter and write a local inspection manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from random import Random

import httpx

from mimeme.source.fetch import Fetcher
from mimeme.source.flip import FlipAdapter, ListingMode
from mimeme.source.http import Http

DEFAULT_OUTPUT = Path("scripts/flip/output/sample.json")


async def run(args: argparse.Namespace) -> int:
    config = {
        "mode": args.mode,
        "max_templates_per_run": args.max_templates,
        "max_meme_pages": args.max_meme_pages,
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
                item
                async for item in FlipAdapter().discover(
                    config, fetcher=fetcher, rng=Random()
                )
            ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "imgflip",
                "mode": args.mode,
                "scraped_at": datetime.now(UTC).isoformat(),
                "items": [item.model_dump(mode="json") for item in items],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    media = [media for item in items for media in item.media]
    blanks = sum(entry.raw_metadata.get("role") == "blank_template" for entry in media)
    examples = sum(entry.raw_metadata.get("role") == "example_meme" for entry in media)
    print(
        f"mode={args.mode} templates={len(items)} media={len(media)} "
        f"blanks={blanks} examples={examples} output={args.output}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ListingMode],
        default=ListingMode.TOP_30_DAYS.value,
    )
    parser.add_argument("--max-templates", type=int, default=3)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-meme-pages", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--impersonate", default="chrome")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
