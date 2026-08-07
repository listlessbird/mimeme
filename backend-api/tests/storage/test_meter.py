from __future__ import annotations

from mimeme import storage
from tests.support.storage import Memory


async def test_reads_and_probes_count_as_class_b() -> None:
    inner = Memory()
    await inner.put_bytes(storage.Object("a.npy"), b"a", content_type="x")
    meter = storage.Meter(inner)

    await meter.read_bytes(storage.Object("a.npy"), max_bytes=16)
    async with meter.read(storage.Object("a.npy")) as chunks:
        async for _ in chunks:
            pass
    await meter.stat(storage.Object("a.npy"))
    await meter.probe()

    counts = meter.counts
    assert (counts.get_object, counts.head_object, counts.head_bucket) == (2, 1, 1)
    assert counts.class_b == 4
    assert counts.class_a == 0


async def test_single_part_put_and_list_pages_count_as_class_a() -> None:
    inner = Memory()
    meter = storage.Meter(inner)

    await meter.put_bytes(storage.Object("small.json"), b"{}", content_type="application/json")
    await meter.put(
        storage.Object("vector.npy"),
        _once(b"0123456789"),
        length=10,
        content_type="application/octet-stream",
        checksum=storage.Checksum.of(b"0123456789"),
    )
    async for _ in meter.list(prefix=""):
        pass

    counts = meter.counts
    assert counts.put_object == 2
    assert counts.list_page == 1
    assert counts.class_a == 3
    assert counts.class_b == 0


async def test_multipart_put_counts_create_parts_and_complete() -> None:
    meter = storage.Meter(Memory(), multipart_threshold=8, multipart_chunk=4)
    data = b"0123456789abcdef"

    await meter.put(
        storage.Object("big.npy"),
        _once(data),
        length=len(data),
        content_type="application/octet-stream",
        checksum=storage.Checksum.of(data),
    )

    counts = meter.counts
    assert (counts.create_multipart, counts.upload_part, counts.complete_multipart) == (1, 4, 1)
    assert counts.put_object == 0
    assert counts.class_a == 6


async def test_deletes_are_free() -> None:
    inner = Memory()
    await inner.put_bytes(storage.Object("gone.npy"), b"x", content_type="x")
    meter = storage.Meter(inner)

    await meter.delete(storage.Object("gone.npy"))

    assert meter.counts == storage.Counts()


async def test_listing_counts_one_page_per_thousand_keys() -> None:
    inner = Memory()
    for position in range(2001):
        await inner.put_bytes(storage.Object(f"k/{position:05d}"), b"x", content_type="x")
    meter = storage.Meter(inner)

    async for _ in meter.list(prefix="k/"):
        pass

    assert meter.counts.list_page == 3


async def _once(data: bytes):  # noqa: ANN202
    yield data
