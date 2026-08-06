from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Self, cast

import httpx
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from mimeme import inference, search, storage
from mimeme.config import ArtifactConfig, MediaConfig, Settings
from mimeme.db import Db
from mimeme.index import ops as index
from mimeme.index.client import Client as IndexClient
from mimeme.index.local import Local as IndexLocal
from mimeme.inference import Client as InferenceClient
from mimeme.ingest.facts import ComputeImages, Images
from mimeme.media import Urls
from mimeme.source.http import Http as SourceHttp


def _storage_config(config: MediaConfig | ArtifactConfig) -> storage.Config:
    return storage.Config(
        endpoint_url=config.s3_endpoint_url,
        region=config.s3_region,
        access_key=config.s3_access_key_id,
        secret_key=config.s3_secret_access_key,
        bucket=config.s3_bucket,
        force_path_style=config.s3_force_path_style,
    )


class Env:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Db,
        temporal: Client,
        media: storage.Store,
        artifacts: storage.Store,
        media_urls: Urls,
        http: httpx.AsyncClient,
        inference: InferenceClient,
        index_client: IndexClient,
        search_client: search.Client,
        image_facts: Images,
        source_http: SourceHttp,
        stack: AsyncExitStack,
    ) -> None:
        self.settings = settings
        self.db = db
        self.temporal = temporal
        self.media = media
        self.artifacts = artifacts
        self.media_urls = media_urls
        self.http = http
        self.inference = inference
        self.index = index_client
        self.search = search_client
        self.image_facts = image_facts
        self.source_http = source_http
        self._stack = stack

    @classmethod
    async def create(cls, settings: Settings) -> Self:
        stack = AsyncExitStack()
        try:
            db = Db(settings.database)
            stack.push_async_callback(db.close)
            temporal = await Client.connect(
                settings.temporal.host,
                namespace=settings.temporal.namespace,
                data_converter=pydantic_data_converter,
            )
            media = await storage.S3.open(_storage_config(settings.media))
            stack.push_async_callback(media.close)
            artifacts = await storage.S3.open(_storage_config(settings.artifacts))
            stack.push_async_callback(artifacts.close)
            media_urls = Urls(settings.media.public_base_url)
            http = httpx.AsyncClient(timeout=settings.compute.request_timeout_s)
            stack.push_async_callback(http.aclose)
            inference_client = inference.create(settings, http)
            stack.push_async_callback(inference_client.close)
            index_client = IndexLocal(
                http,
                base_url=settings.compute.gateway_url,
                poll_interval_s=settings.compute.poll_interval_s,
            )
            stack.push_async_callback(index_client.close)
            search_client = search.create(settings, http)
            stack.push_async_callback(search_client.close)
            await index.reconcile(
                db,
                artifacts,
                cast(search.Activation, search_client),
            )
            image_facts = ComputeImages(http, base_url=settings.compute.gateway_url)
            return cls(
                settings=settings,
                db=db,
                temporal=temporal,
                media=media,
                artifacts=artifacts,
                media_urls=media_urls,
                http=http,
                inference=inference_client,
                index_client=index_client,
                search_client=search_client,
                image_facts=image_facts,
                source_http=SourceHttp(http),
                stack=stack,
            )
        except BaseException:
            await stack.aclose()
            raise

    async def aclose(self) -> None:
        await self._stack.aclose()
