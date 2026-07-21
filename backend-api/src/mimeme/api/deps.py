from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from temporalio.client import Client

from mimeme import storage
from mimeme.activities.indexing import FaissIndexManager
from mimeme.db import Db
from mimeme.env import Env
from mimeme.shared.config import Settings
from mimeme.shared.services.media_url import MediaUrlResolver


def get_env(request: Request) -> Env:
    return request.app.state.env


EnvDep = Annotated[Env, Depends(get_env)]


def get_settings(request: Request) -> Settings:
    return request.app.state.env.settings


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_db(request: Request) -> Db:
    return request.app.state.env.db


DbDep = Annotated[Db, Depends(get_db)]


def get_media_storage(request: Request) -> storage.Store:
    return request.app.state.env.media


MediaStorageDep = Annotated[storage.Store, Depends(get_media_storage)]


def get_artifact_storage(request: Request) -> storage.Store:
    return request.app.state.env.artifacts


ArtifactStorageDep = Annotated[storage.Store, Depends(get_artifact_storage)]


def get_media_url_resolver(request: Request) -> MediaUrlResolver:
    return request.app.state.env.media_urls


MediaUrlResolverDep = Annotated[MediaUrlResolver, Depends(get_media_url_resolver)]


def get_index_manager(request: Request) -> FaissIndexManager:
    return request.app.state.env.index_manager


IndexManagerDep = Annotated[FaissIndexManager, Depends(get_index_manager)]


async def get_temporal_client(request: Request) -> Client:
    return request.app.state.env.temporal


TemporalClientDep = Annotated[Client, Depends(get_temporal_client)]
