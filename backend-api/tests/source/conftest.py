from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncConnection

from mimeme.source.http import Http
from mimeme.source.model import FetchRequest, RawResponse
from tests.job.conftest import PoolDb, SavepointDb, _async_url


@pytest.fixture()
def db(async_db_connection: AsyncConnection) -> SavepointDb:
    return SavepointDb(async_db_connection)


@pytest.fixture()
async def pool_db(db_engine: Engine):
    if db_engine.dialect.name != "postgresql":
        pytest.skip("uniqueness/locking tests require PostgreSQL")
    pool = PoolDb(_async_url(db_engine))
    yield pool
    await pool.close()


class FakeHttp:
    """Stands in for ``source.Http``. Maps request URL to a canned
    ``RawResponse`` or an exception to raise."""

    def __init__(self) -> None:
        self._by_url: dict[str, RawResponse | Exception] = {}
        self._default: RawResponse | Exception | None = None
        self.calls: list[FetchRequest] = []

    def set(self, url: str, outcome: RawResponse | Exception) -> None:
        self._by_url[url] = outcome

    def default(self, outcome: RawResponse | Exception) -> None:
        self._default = outcome

    async def fetch(self, request: FetchRequest) -> RawResponse:
        self.calls.append(request)
        outcome = self._by_url.get(request.url, self._default)
        if outcome is None:
            raise AssertionError(f"no fake response configured for {request.url}")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class FakeEnv:
    db: object
    source_http: FakeHttp = field(default_factory=FakeHttp)


def real_http(client) -> Http:
    return Http(client)


def meme_response(*post_ids: str) -> RawResponse:
    memes = [
        {
            "postLink": f"https://reddit.com/r/memes/comments/{pid}/x",
            "url": f"https://i.redd.it/{pid}.jpg",
            "title": f"meme {pid}",
            "nsfw": False,
            "ups": 500,
        }
        for pid in post_ids
    ]
    return RawResponse(success=True, status_code=200, raw={"memes": memes})
