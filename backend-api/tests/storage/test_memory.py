from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from mimeme import storage
from tests.support.storage import Memory


def _iter(*chunks: bytes) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    return gen()


async def _drain(store: storage.Store, obj: storage.Object) -> bytes:
    buffer = bytearray()
    async with store.read(obj) as chunks:
        async for chunk in chunks:
            buffer += chunk
    return bytes(buffer)


@pytest.fixture()
def store() -> Memory:
    return Memory(put_bytes_max=32)


async def test_put_streams_and_returns_info(store: Memory) -> None:
    obj = storage.Object("images/a/b/x.jpg")
    data = b"chunk-one" + b"chunk-two"
    info = await store.put(
        obj,
        _iter(b"chunk-one", b"chunk-two"),
        length=len(data),
        content_type="image/jpeg",
        checksum=storage.Checksum.of(data),
    )
    assert info.object == obj
    assert info.length == len(data)
    assert info.content_type == "image/jpeg"
    assert info.checksum == storage.Checksum.of(data)
    assert await _drain(store, obj) == data


async def test_put_rejects_length_mismatch(store: Memory) -> None:
    obj = storage.Object("a/b")
    data = b"payload"
    with pytest.raises(storage.Integrity):
        await store.put(
            obj,
            _iter(data),
            length=99,
            content_type="text/plain",
            checksum=storage.Checksum.of(data),
        )
    assert await store.stat(obj) is None


async def test_put_rejects_checksum_mismatch(store: Memory) -> None:
    obj = storage.Object("a/b")
    data = b"payload"
    with pytest.raises(storage.Integrity):
        await store.put(
            obj,
            _iter(data),
            length=len(data),
            content_type="text/plain",
            checksum=storage.Checksum.of(b"different"),
        )


async def test_put_bytes_enforces_small_object_limit(store: Memory) -> None:
    obj = storage.Object("small/json")
    with pytest.raises(storage.Invalid):
        await store.put_bytes(obj, b"x" * 64, content_type="application/json")
    info = await store.put_bytes(obj, b"{}", content_type="application/json")
    assert info.length == 2


async def test_read_bytes_enforces_max(store: Memory) -> None:
    obj = storage.Object("a/b")
    await store.put_bytes(obj, b"0123456789", content_type="text/plain")
    assert await store.read_bytes(obj, max_bytes=10) == b"0123456789"
    with pytest.raises(storage.Invalid):
        await store.read_bytes(obj, max_bytes=4)


async def test_missing_reads_raise_missing(store: Memory) -> None:
    obj = storage.Object("nope/here")
    with pytest.raises(storage.Missing):
        await store.read_bytes(obj, max_bytes=10)
    with pytest.raises(storage.Missing):
        await _drain(store, obj)


async def test_stat_returns_none_for_absent(store: Memory) -> None:
    assert await store.stat(storage.Object("nope/here")) is None


async def test_delete_is_idempotent(store: Memory) -> None:
    obj = storage.Object("a/b")
    await store.put_bytes(obj, b"hi", content_type="text/plain")
    await store.delete(obj)
    await store.delete(obj)
    assert await store.stat(obj) is None


async def test_list_filters_by_prefix_and_orders(store: Memory) -> None:
    for key in ["images/b", "images/a", "embeddings/z"]:
        await store.put_bytes(storage.Object(key), b"x", content_type="text/plain")
    keys = [info.object.key async for info in store.list(prefix="images/")]
    assert keys == ["images/a", "images/b"]
    all_keys = [info.object.key async for info in store.list()]
    assert all_keys == ["embeddings/z", "images/a", "images/b"]


async def test_one_shot_input_is_not_replayed(store: Memory) -> None:
    obj = storage.Object("a/b")
    body = _iter(b"first")
    await store.put(
        obj, body, length=5, content_type="text/plain", checksum=storage.Checksum.of(b"first")
    )
    # The consumed iterable is exhausted; a retry must supply a fresh one.
    fresh = _iter(b"first")
    info = await store.put(
        obj, fresh, length=5, content_type="text/plain", checksum=storage.Checksum.of(b"first")
    )
    assert info.length == 5


async def test_close_marks_closed(store: Memory) -> None:
    await store.probe()
    await store.close()
    assert store.closed is True
