from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, Protocol, cast

from aiobotocore.config import AioConfig
from aiobotocore.session import ClientCreatorContext, get_session

from shared.services.storage import S3Config

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client

    class S3Session(Protocol):
        def create_client(
            self, service_name: Literal["s3"], **kwargs: object
        ) -> ClientCreatorContext[S3Client]: ...


_clients: dict[tuple[asyncio.AbstractEventLoop, S3Config], S3Client] = {}
_client_locks: dict[tuple[asyncio.AbstractEventLoop, S3Config], asyncio.Lock] = {}


async def get_loop_client(config: S3Config) -> S3Client:
    loop = asyncio.get_running_loop()
    cache_key = (loop, config)
    if (client := _clients.get(cache_key)) is not None:
        return client

    lock = _client_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        if (client := _clients.get(cache_key)) is None:
            session = cast("S3Session", get_session())
            ctx = session.create_client(
                "s3",
                region_name=config.region,
                endpoint_url=config.endpoint_url,
                aws_access_key_id=config.access_key,
                aws_secret_access_key=config.secret_key,
                config=AioConfig(
                    s3={"addressing_style": "path" if config.force_path_style else "auto"},
                    signature_version="s3v4",
                ),
            )
            client = await ctx.__aenter__()
            _clients[cache_key] = client
    return client
