from __future__ import annotations

import asyncio
import time

from mimeme import search
from mimeme.search.run import Projection, run


class _SlowClient:
    async def query(
        self, query: search.Query, *, count: int, cursor: str | None = None
    ) -> search.Batch:
        await asyncio.sleep(0.2)
        return search.Batch(candidates=[], exhausted=True, version="v1")

    async def status(self) -> search.Status:
        return search.Status(ready=True, serving_version="v1")

    async def close(self) -> None:
        pass


class _Rows:
    async def fetch(self, image_ids: list[int]) -> dict[int, Projection]:
        return {}


class _Urls:
    def resolve(self, key: str) -> str:
        return key


async def test_concurrent_searches_do_not_serialize_the_api_event_loop() -> None:
    started = time.perf_counter()
    await asyncio.gather(
        *(
            run(
                search.Query(text=f"query {index}"),
                client=_SlowClient(),
                rows=_Rows(),
                media_urls=_Urls(),
            )
            for index in range(4)
        )
    )

    assert time.perf_counter() - started < 0.4
