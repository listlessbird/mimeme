from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncConnection
from tests.job.conftest import SavepointDb


@pytest.fixture()
def index_db(async_db_connection: AsyncConnection) -> SavepointDb:
    return SavepointDb(async_db_connection)
