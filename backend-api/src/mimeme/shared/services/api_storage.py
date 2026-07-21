from __future__ import annotations

from typing import BinaryIO, Protocol

from botocore.client import ClientError

from mimeme.shared.services.api_storage_client import get_loop_client
from mimeme.shared.services.storage import S3Config


class ApiStorage(Protocol):
    async def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...


class AsyncApiStorage:
    def __init__(self, config: S3Config) -> None:
        self._config = config

    @property
    def bucket(self) -> str:
        return self._config.bucket

    async def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str:
        client = await get_loop_client(self._config)
        response = await client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )
        return response.get("ETag", "").strip('"')

    async def delete(self, key: str) -> None:
        client = await get_loop_client(self._config)
        await client.delete_object(Bucket=self.bucket, Key=key)

    async def exists(self, key: str) -> bool:
        client = await get_loop_client(self._config)
        try:
            await client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    async def bucket_exists(self) -> bool:
        client = await get_loop_client(self._config)
        try:
            await client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError:
            return False
