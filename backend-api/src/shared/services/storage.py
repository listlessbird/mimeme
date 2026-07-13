from __future__ import annotations

import io
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, ClassVar

import boto3
import numpy as np
from boto3.s3.transfer import TransferConfig
from botocore.client import ClientError
from botocore.config import Config as BotoConfig
from pydantic import BaseModel

from shared.config import Settings, settings

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client


class S3Config(BaseModel):
    endpoint_url: str
    region: str
    access_key: str
    secret_key: str
    bucket: str
    force_path_style: bool

    model_config = {"frozen": True}


def get_media_s3_config(app_settings: Settings = settings) -> S3Config:
    return S3Config(
        endpoint_url=app_settings.media_s3_endpoint_url,
        region=app_settings.media_s3_region,
        access_key=app_settings.media_s3_access_key_id,
        secret_key=app_settings.media_s3_secret_access_key,
        bucket=app_settings.media_s3_bucket,
        force_path_style=app_settings.media_s3_force_path_style,
    )


def get_artifact_s3_config(app_settings: Settings = settings) -> S3Config:
    return S3Config(
        endpoint_url=app_settings.artifact_s3_endpoint_url,
        region=app_settings.artifact_s3_region,
        access_key=app_settings.artifact_s3_access_key_id,
        secret_key=app_settings.artifact_s3_secret_access_key,
        bucket=app_settings.artifact_s3_bucket,
        force_path_style=app_settings.artifact_s3_force_path_style,
    )


class StorageService:
    IMAGES_PREFIX = "images"
    EMBEGGINGS_PREFIX = "embeddings"
    INDEXES_PREFIX: ClassVar[str] = "indexes"

    def __init__(self, config: S3Config) -> None:
        self._config = config
        self._client: S3Client | None = None
        self._transfer_config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,
            max_concurrency=8,
            multipart_chunksize=8 * 1024 * 1024,
            use_threads=True,
        )

    @property
    def client(self) -> S3Client:
        if self._client is None:
            session = boto3.Session(
                aws_access_key_id=self._config.access_key,
                aws_secret_access_key=self._config.secret_key,
                region_name=self._config.region,
            )
            self._client = session.client(
                "s3",
                endpoint_url=self._config.endpoint_url,
                config=BotoConfig(
                    s3={"addressing_style": ("path" if self._config.force_path_style else "auto")},
                    signature_version="s3v4",
                ),
            )
        return self._client

    @property
    def bucket(self) -> str:
        return self._config.bucket

    def bucket_exists(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError:
            return False

    def build_image_key(self, sha256: str, dataset: str | None, extension: str) -> str:
        ext = extension.lower().lstrip(".")
        source = dataset if dataset else "api-ingested"
        return f"{self.IMAGES_PREFIX}/{source}/{sha256[:2]}/{sha256[2:4]}/{sha256}.{ext}"

    def build_embedding_key(self, sha256: str, model_name: str, dataset: str | None) -> str:
        source = dataset if dataset else "api-ingested"
        model_slug = model_name.replace("/", "_")
        return f"{self.EMBEGGINGS_PREFIX}/{model_slug}/{source}/{sha256}.npy"

    def build_index_key(self, version: str, filename: str) -> str:
        return f"{self.INDEXES_PREFIX}/{version}/{filename}"

    def upload_file(self, local_path: Path, key: str, content_type: str | None = None) -> str:
        ct = content_type or self._guess_content_type(local_path)

        self.client.upload_file(
            Filename=str(local_path),
            Bucket=self.bucket,
            Key=key,
            Config=self._transfer_config,
            ExtraArgs={"ContentType": ct},
        )

        return self._get_etag(key) or ""

    def upload_bytes(
        self, data: bytes | BinaryIO, key: str, content_type: str = "application/octet-stream"
    ) -> str:
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        self.client.upload_fileobj(
            Fileobj=data, Bucket=self.bucket, Key=key, ExtraArgs={"ContentType": content_type}
        )
        return self._get_etag(key) or ""

    def upload_numpy(self, arr: np.ndarray, key: str) -> str:
        buffer = io.BytesIO()
        np.save(buffer, arr)
        buffer.seek(0)
        return self.upload_bytes(buffer, key, "application/octet-stream")

    def download_file(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(Bucket=self.bucket, Key=key, Filename=str(local_path))

    def download_bytes(self, key: str) -> bytes:
        buffer = io.BytesIO()
        self.client.download_fileobj(Bucket=self.bucket, Key=key, Fileobj=buffer)
        buffer.seek(0)
        return buffer.read()

    def download_numpy(self, key: str) -> np.ndarray:
        buffer = io.BytesIO()
        self.client.download_fileobj(Bucket=self.bucket, Key=key, Fileobj=buffer)
        buffer.seek(0)
        return np.load(buffer)

    def download_to_temp(self, key: str, suffix: str = "") -> Path:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            self.client.download_file(Bucket=self.bucket, Key=key, Filename=path)
            return Path(path)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def list_objects(self, prefix: str) -> list[tuple[str, int]]:
        result: list[tuple[str, int]] = []

        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if "Key" in obj and "Size" in obj:
                    result.append((obj["Key"], obj["Size"]))

        return result

    def _guess_content_type(self, path: Path) -> str:
        ext = path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".json": "application/json",
            ".npy": "application/octet-stream",
            ".faiss": "application/octet-stream",
        }.get(ext, "application/octet-stream")

    def _get_etag(self, key: str) -> str | None:
        try:
            resp = self.client.head_object(Bucket=self.bucket, Key=key)
            return resp.get("ETag", "").strip('"')
        except ClientError:
            return None


@lru_cache(maxsize=1)
def get_media_storage_service() -> StorageService:
    return StorageService(get_media_s3_config())


@lru_cache(maxsize=1)
def get_artifact_storage_service() -> StorageService:
    return StorageService(get_artifact_s3_config())
