from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiobotocore.config import AioConfig
from aiobotocore.session import get_session

from shared.services.storage import get_s3_config

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client

_clients: dict[asyncio.AbstractEventLoop, S3Client] = {}
_client_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}


async def get_loop_client() -> S3Client:
    loop = asyncio.get_running_loop()
    if (client := _clients.get(loop)) is not None:
        return client

    lock = _client_locks.setdefault(loop, asyncio.Lock())
    async with lock:
        if (client := _clients.get(loop)) is None:
            config = get_s3_config()
            ctx = get_session().create_client(
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
            _clients[loop] = client
    return client
