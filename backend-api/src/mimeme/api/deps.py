from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from mimeme.activities.indexing import FaissIndexManager
from mimeme.shared.config import settings
from mimeme.shared.services.api_storage import ApiStorage, AsyncApiStorage
from mimeme.shared.services.media_url import MediaUrlResolver
from mimeme.shared.services.storage import (
    get_artifact_s3_config,
    get_media_s3_config,
)


def get_media_storage() -> ApiStorage:
    return AsyncApiStorage(get_media_s3_config())


MediaStorageDep = Annotated[ApiStorage, Depends(get_media_storage)]


def get_artifact_storage() -> ApiStorage:
    return AsyncApiStorage(get_artifact_s3_config())


ArtifactStorageDep = Annotated[ApiStorage, Depends(get_artifact_storage)]


@lru_cache(maxsize=1)
def get_media_url_resolver() -> MediaUrlResolver:
    return MediaUrlResolver(settings.media.public_base_url)


MediaUrlResolverDep = Annotated[MediaUrlResolver, Depends(get_media_url_resolver)]


@lru_cache(maxsize=1)
def get_index_manager() -> FaissIndexManager:
    return FaissIndexManager.get_instance()


IndexManagerDep = Annotated[FaissIndexManager, Depends(get_index_manager)]


_temporal_client: Client | None = None


async def get_temporal_client() -> Client:
    global _temporal_client
    if _temporal_client is None:
        _temporal_client = await Client.connect(
            settings.temporal.host,
            data_converter=pydantic_data_converter,
        )
    return _temporal_client


TemporalClientDep = Annotated[Client, Depends(get_temporal_client)]
