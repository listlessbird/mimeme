from __future__ import annotations

import time
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from mimeme.search.client import Client
from mimeme.search.error import Failed, Stale
from mimeme.search.model import Page, Query, Result


class Projection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    sha256: str
    media_key: str | None
    caption: str | None
    ocr_text: str | None
    width: int | None
    height: int | None


class Rows(Protocol):
    async def fetch(self, image_ids: list[int]) -> dict[int, Projection]: ...


class Urls(Protocol):
    def resolve(self, key: str) -> str: ...


async def run(
    query: Query,
    *,
    client: Client,
    rows: Rows,
    media_urls: Urls,
) -> Page:
    started = time.perf_counter()
    required = query.offset + query.limit
    cursor: str | None = None
    seen_cursors: set[str] = set()
    seen_ids: set[int] = set()
    ranked: list[Result] = []
    version: str | None = None

    while len(ranked) < required:
        count = min(
            1000,
            required if cursor is None else max(32, (required - len(ranked)) * 2),
        )
        batch = await client.query(query, count=count, cursor=cursor)
        if version is None:
            version = batch.version
        elif batch.version != version:
            raise Stale(f"search index changed from {version!r} to {batch.version!r}")

        candidates = [
            candidate
            for candidate in batch.candidates
            if candidate.image_id not in seen_ids and candidate.image_id != query.similar_image_id
        ]
        seen_ids.update(candidate.image_id for candidate in batch.candidates)
        found = await rows.fetch([candidate.image_id for candidate in candidates])
        for candidate in candidates:
            if len(ranked) >= required:
                break
            row = found.get(candidate.image_id)
            if row is None:
                continue
            ranked.append(
                Result(
                    id=row.id,
                    sha256=row.sha256,
                    score=candidate.score,
                    url=media_urls.resolve(row.media_key) if row.media_key else None,
                    caption=row.caption,
                    ocr_text=row.ocr_text,
                    width=row.width,
                    height=row.height,
                )
            )

        if batch.exhausted:
            break
        assert batch.cursor is not None
        if batch.cursor in seen_cursors or batch.cursor == cursor:
            raise Failed("search compute returned a repeated cursor")
        seen_cursors.add(batch.cursor)
        cursor = batch.cursor

    label = query.text if query.text is not None else f"similar_to:{query.similar_image_id}"
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return Page(
        query=label,
        results=ranked[query.offset : required],
        total=len(ranked),
        limit=query.limit,
        offset=query.offset,
        search_time_ms=elapsed_ms,
        index_version=version,
    )
