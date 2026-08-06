from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import Depends, Request
from temporalio.client import Client

from mimeme import inference, search, storage
from mimeme.config import Settings
from mimeme.db import Db
from mimeme.env import Env
from mimeme.media import Urls


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


def get_search(request: Request) -> search.Client:
    return request.app.state.env.search


SearchDep = Annotated[search.Client, Depends(get_search)]


def get_inference(request: Request) -> inference.Client:
    return request.app.state.env.inference


InferenceDep = Annotated[inference.Client, Depends(get_inference)]


def get_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.env.http


HttpDep = Annotated[httpx.AsyncClient, Depends(get_http)]


def get_media_url_resolver(request: Request) -> Urls:
    return request.app.state.env.media_urls


UrlsDep = Annotated[Urls, Depends(get_media_url_resolver)]


async def get_temporal_client(request: Request) -> Client:
    return request.app.state.env.temporal


TemporalClientDep = Annotated[Client, Depends(get_temporal_client)]
