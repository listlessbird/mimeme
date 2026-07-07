from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from activities.indexing import FaissIndexManager
from shared.config import settings
from shared.db import get_db
from shared.services.api_storage import ApiStorage, BotoApiStorage
from shared.services.storage import get_storage_service

DbSession = Annotated[Session, Depends(get_db)]


def get_storage() -> ApiStorage:
    return BotoApiStorage(get_storage_service())


StorageDep = Annotated[ApiStorage, Depends(get_storage)]


@lru_cache(maxsize=1)
def get_index_manager() -> FaissIndexManager:
    return FaissIndexManager.get_instance()


IndexManagerDep = Annotated[FaissIndexManager, Depends(get_index_manager)]


_temporal_client: Client | None = None


async def get_temporal_client() -> Client:
    global _temporal_client
    if _temporal_client is None:
        _temporal_client = await Client.connect(
            settings.temporal_host,
            data_converter=pydantic_data_converter,
        )
    return _temporal_client


TemporalClientDep = Annotated[Client, Depends(get_temporal_client)]
