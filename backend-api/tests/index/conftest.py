from __future__ import annotations

import pytest
from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncConnection
from tests.job.conftest import PoolDb, SavepointDb, _async_url


@pytest.fixture()
def index_db(async_db_connection: AsyncConnection) -> SavepointDb:
    return SavepointDb(async_db_connection)


@pytest.fixture()
async def pool_db(db_engine: Engine):
    if db_engine.dialect.name != "postgresql":
        pytest.skip("advisory-lock tests require PostgreSQL")
    pool = PoolDb(_async_url(db_engine))
    yield pool
    await pool.close()
