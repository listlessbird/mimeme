from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

from sqlalchemy import text

from mimeme.config import Settings
from mimeme.db import Db

SAMPLE_SQL = """
SELECT i.id, i.width, i.height, a.ocr_text, a.caption_text, a.tags
FROM images i
JOIN annotations a ON a.image_id = i.id
JOIN processing p ON p.image_id = i.id
WHERE p.embed_status = 'DONE'
  AND i.dataset = :dataset
  AND (a.ocr_text IS NOT NULL OR a.caption_text IS NOT NULL OR a.tags IS NOT NULL)
"""


def aspect_bucket(width: int | None, height: int | None) -> str:
    if not width or not height:
        return "unknown"
    ratio = width / height
    if ratio < 0.8:
        return "tall"
    if ratio > 1.25:
        return "wide"
    return "square"


def semantic_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        return json.loads(raw).get("semantic_tags", [])[:8]
    except (json.JSONDecodeError, AttributeError):
        return []


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="ImgFlip575K Memes Dataset")
    parser.add_argument("--out", type=Path, default=Path("evals/eval_queries_input.jsonl"))
    args = parser.parse_args()

    db = Db(Settings().database)
    try:
        async with db.read_session() as session:
            rows = (await session.execute(text(SAMPLE_SQL), {"dataset": args.dataset})).all()
    finally:
        await db.close()

    rng = random.Random(args.seed)
    buckets: dict[str, list[tuple[Any, ...]]] = {}
    for row in rows:
        buckets.setdefault(aspect_bucket(row[1], row[2]), []).append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    picked: list[tuple[Any, ...]] = []
    order = sorted(buckets, key=lambda b: len(buckets[b]))
    while len(picked) < args.n and any(buckets.values()):
        for bucket in order:
            if buckets[bucket] and len(picked) < args.n:
                picked.append(buckets[bucket].pop())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for image_id, width, height, ocr, caption, tags in sorted(picked):
            record = {
                "image_id": image_id,
                "aspect": aspect_bucket(width, height),
                "ocr_text": (ocr or "").strip()[:400],
                "caption_text": (caption or "").strip()[:400],
                "tags": semantic_tags(tags),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"sampled {len(picked)} of {len(rows)} candidates → {args.out}")
    counts: dict[str, int] = {}
    for row in picked:
        counts[aspect_bucket(row[1], row[2])] = counts.get(aspect_bucket(row[1], row[2]), 0) + 1
    print(f"aspect mix: {counts}")


if __name__ == "__main__":
    asyncio.run(main())
