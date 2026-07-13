from __future__ import annotations

from typing import BinaryIO, Protocol

from botocore.client import ClientError

from shared.services.api_storage_client import get_loop_client
from shared.services.storage import get_s3_config


class ApiStorage(Protocol):
    def presign(self, key: str, expiration: int = 3600) -> str: ...
    async def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...


class ApiStorageProbe(Protocol):
    @property
    def bucket(self) -> str: ...
    def ensure_bucket_exists(self) -> None: ...
    def bucket_exists(self) -> bool: ...


class ApiStorageAdapter(Protocol):
    @property
    def bucket(self) -> str: ...
    def generate_presigned_url(self, key: str, expiration: int = 3600) -> str: ...
    def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    def ensure_bucket_exists(self) -> None: ...
    def bucket_exists(self) -> bool: ...


class AsyncApiStorage:
    def __init__(self, presigner: ApiStorageAdapter) -> None:
        self._presigner = presigner

    def presign(self, key: str, expiration: int = 3600) -> str:
        return self._presigner.generate_presigned_url(key, expiration=expiration)

    @property
    def bucket(self) -> str:
        return get_s3_config().bucket

    async def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str:
        client = await get_loop_client()
        response = await client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )
        return response.get("ETag", "").strip('"')

    async def delete(self, key: str) -> None:
        client = await get_loop_client()
        await client.delete_object(Bucket=self.bucket, Key=key)

    async def exists(self, key: str) -> bool:
        client = await get_loop_client()
        try:
            await client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    async def bucket_exists(self) -> bool:
        client = await get_loop_client()
        try:
            await client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError:
            return False


class BotoApiStorage:
    def __init__(self, storage: ApiStorageAdapter) -> None:
        self._storage = storage

    def presign(self, key: str, expiration: int = 3600) -> str:
        return self._storage.generate_presigned_url(key, expiration=expiration)

    @property
    def bucket(self) -> str:
        return self._storage.bucket

    async def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str:
        return self._storage.upload_bytes(data, key, content_type)

    async def delete(self, key: str) -> None:
        self._storage.delete(key)

    async def exists(self, key: str) -> bool:
        return self._storage.exists(key)

    def ensure_bucket_exists(self) -> None:
        self._storage.ensure_bucket_exists()

    def bucket_exists(self) -> bool:
        return self._storage.bucket_exists()
