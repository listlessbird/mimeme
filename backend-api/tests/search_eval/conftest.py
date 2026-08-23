from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncConnection, async_sessionmaker

from mimeme.db import Db


class SavepointDb(Db):
    def __init__(self, connection: AsyncConnection) -> None:
        self._engine = None  # type: ignore[assignment]
        self._sessions = async_sessionmaker(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )

    async def close(self) -> None:
        return None


@pytest.fixture()
def eval_db(async_db_connection: AsyncConnection) -> SavepointDb:
    return SavepointDb(async_db_connection)
