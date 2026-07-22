from __future__ import annotations

from typing import Self

import httpx
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from mimeme import inference, storage
from mimeme.db import Db
from mimeme.inference import Client as InferenceClient
from mimeme.ingest.facts import ComputeImages, Images
from mimeme.shared.config import ArtifactConfig, MediaConfig, Settings
from mimeme.shared.services.media_url import MediaUrlResolver
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
        media_urls: MediaUrlResolver,
        http: httpx.AsyncClient,
        inference: InferenceClient,
        image_facts: Images,
        source_http: SourceHttp,
    ) -> None:
        self.settings = settings
        self.db = db
        self.temporal = temporal
        self.media = media
        self.artifacts = artifacts
        self.media_urls = media_urls
        self.http = http
        self.inference = inference
        self.image_facts = image_facts
        self.source_http = source_http

    @classmethod
    async def create(cls, settings: Settings) -> Self:
        db = Db(settings.database)
        temporal = await Client.connect(
            settings.temporal.host,
            data_converter=pydantic_data_converter,
        )
        media = await storage.S3.open(_storage_config(settings.media))
        artifacts = await storage.S3.open(_storage_config(settings.artifacts))
        media_urls = MediaUrlResolver(settings.media.public_base_url)
        http = httpx.AsyncClient(timeout=settings.compute.request_timeout_s)
        inference_client = inference.create(settings, http)
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
            image_facts=image_facts,
            source_http=SourceHttp(http),
        )

    async def aclose(self) -> None:
        await self.inference.close()
        await self.http.aclose()
        await self.artifacts.close()
        await self.media.close()
        await self.db.close()
