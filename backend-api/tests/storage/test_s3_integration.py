from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import pytest
from pydantic import SecretStr

from mimeme import storage

pytestmark = pytest.mark.storage_integration

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "minio", "::1"}
_SAFE_PREFIX = "mimeme-test-"


def _enabled() -> bool:
    return os.environ.get("STORAGE_INTEGRATION_TARGET") == "minio-local"


def _require_loopback(endpoint: str) -> None:
    host = urlsplit(endpoint).hostname or ""
    if host not in _LOOPBACK_HOSTS:
        raise RuntimeError(f"refusing to run storage integration against non-local host {host!r}")


def _iter(*chunks: bytes) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    return gen()


@pytest.fixture()
def run_prefix() -> str:
    return f"{_SAFE_PREFIX}{uuid.uuid4().hex}"


@pytest.fixture()
async def s3(run_prefix: str) -> AsyncIterator[tuple[storage.S3, str]]:
    if not _enabled():
        pytest.skip("set STORAGE_INTEGRATION_TARGET=minio-local to run MinIO contract tests")

    endpoint = os.environ.get("MINIO_ENDPOINT_URL", "http://127.0.0.1:9000")
    _require_loopback(endpoint)
    bucket = os.environ.get("MINIO_TEST_BUCKET", "mimeme-media")

    config = storage.Config(
        endpoint_url=endpoint,
        region=os.environ.get("MINIO_REGION", "us-east-1"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=SecretStr(os.environ.get("MINIO_SECRET_KEY", "minioadmin")),
        bucket=bucket,
        force_path_style=True,
        multipart_threshold=5 * 1024 * 1024,
        multipart_chunk=5 * 1024 * 1024,
    )
    store = await storage.S3.open(config)
    try:
        yield store, run_prefix
    finally:
        async for info in store.list(prefix=run_prefix):
            await store.delete(info.object)
        await store.close()


def _guard_prefix(key: str) -> storage.Object:
    if not key.startswith(_SAFE_PREFIX):
        raise RuntimeError(f"refusing to write outside the guarded prefix: {key!r}")
    return storage.Object(key)


async def test_put_get_stat_delete(s3: tuple[storage.S3, str]) -> None:
    store, prefix = s3
    obj = _guard_prefix(f"{prefix}/small.txt")
    data = b"hello minio"
    info = await store.put(
        obj,
        _iter(data),
        length=len(data),
        content_type="text/plain",
        checksum=storage.Checksum.of(data),
    )
    assert info.length == len(data)

    assert await store.read_bytes(obj, max_bytes=1024) == data
    stat = await store.stat(obj)
    assert stat is not None
    assert stat.length == len(data)
    assert stat.checksum == storage.Checksum.of(data)

    await store.delete(obj)
    await store.delete(obj)
    assert await store.stat(obj) is None


async def test_streamed_multipart_upload_and_download(s3: tuple[storage.S3, str]) -> None:
    store, prefix = s3
    obj = _guard_prefix(f"{prefix}/big.bin")
    part = b"a" * (1024 * 1024)
    chunks = [part] * 12
    data = b"".join(chunks)
    info = await store.put(
        obj,
        _iter(*chunks),
        length=len(data),
        content_type="application/octet-stream",
        checksum=storage.Checksum.of(data),
    )
    assert info.length == len(data)

    downloaded = bytearray()
    async with store.read(obj) as stream:
        async for chunk in stream:
            downloaded += chunk
    assert bytes(downloaded) == data


async def test_list_paginates_full_prefix(s3: tuple[storage.S3, str]) -> None:
    store, prefix = s3
    expected = sorted(f"{prefix}/page/{i:04d}.txt" for i in range(2500))
    for key in expected:
        await store.put_bytes(_guard_prefix(key), b"x", content_type="text/plain")

    keys = [info.object.key async for info in store.list(prefix=f"{prefix}/page/")]
    assert keys == expected


async def test_missing_object_raises(s3: tuple[storage.S3, str]) -> None:
    store, prefix = s3
    obj = _guard_prefix(f"{prefix}/absent.txt")
    with pytest.raises(storage.Missing):
        await store.read_bytes(obj, max_bytes=16)


async def test_nonlocal_endpoint_is_refused() -> None:
    with pytest.raises(RuntimeError):
        _require_loopback("https://s3.amazonaws.com")
    with pytest.raises(RuntimeError):
        _guard_prefix("images/not-guarded.txt")
