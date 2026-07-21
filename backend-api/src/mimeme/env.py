from __future__ import annotations

from typing import Self

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from mimeme.db import Db
from mimeme.shared.config import Settings


class Env:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Db,
        temporal: Client,
    ) -> None:
        self.settings = settings
        self.db = db
        self.temporal = temporal

    @classmethod
    async def create(cls, settings: Settings) -> Self:
        db = Db(settings.database)
        temporal = await Client.connect(
            settings.temporal.host,
            data_converter=pydantic_data_converter,
        )
        return cls(settings=settings, db=db, temporal=temporal)

    async def aclose(self) -> None:
        await self.db.close()
