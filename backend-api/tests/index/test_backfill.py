from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session
from tests.factories import create_image, create_processing
from tests.job.conftest import SavepointDb
from tests.support.storage import Memory

from mimeme import storage
from mimeme.db.schema import Processing, ProcessingStatus
from mimeme.index import ops

_MODEL = "google/siglip2-base"
_PREFIX = "embeddings/google_siglip2-base"


def _embedded(session: Session, *, key: str) -> int:
    image = create_image(session=session)
    processing = create_processing(session=session, image=image)
    processing.embed_status = ProcessingStatus.DONE
    processing.embed_model = _MODEL
    processing.embed_dim = 2
    processing.embed_s3_key = key
    processing.embed_text_present = None
    session.flush()
    return image.id


async def _presence(db: SavepointDb) -> dict[str, bool | None]:
    async with db.read_session() as session:
        rows = (
            await session.execute(select(Processing.embed_s3_key, Processing.embed_text_present))
        ).all()
    return {str(row.embed_s3_key): row.embed_text_present for row in rows}


async def test_backfill_marks_presence_from_one_listing_pass(
    index_db: SavepointDb, run_sync_seed
) -> None:
    def seed(session: Session) -> None:
        _embedded(session, key=f"{_PREFIX}/api-ingested/aa.npy")
        _embedded(session, key=f"{_PREFIX}/api-ingested/bb.npy")

    await run_sync_seed(seed)
    artifacts = Memory()
    for key in (
        f"{_PREFIX}/api-ingested/aa.npy",
        f"{_PREFIX}/api-ingested/aa_text.npy",
        f"{_PREFIX}/api-ingested/bb.npy",
    ):
        await artifacts.put_bytes(storage.Object(key), b"v", content_type="x")
    meter = storage.Meter(artifacts)

    result = await ops.backfill_text_presence(index_db, meter, model=_MODEL)

    assert result.text_objects == 1
    assert result.marked_present == 1
    assert result.marked_absent == 1
    assert await _presence(index_db) == {
        f"{_PREFIX}/api-ingested/aa.npy": True,
        f"{_PREFIX}/api-ingested/bb.npy": False,
    }
    assert meter.counts.class_b == 0


async def test_backfill_is_idempotent_and_leaves_settled_rows_alone(
    index_db: SavepointDb, run_sync_seed
) -> None:
    def seed(session: Session) -> None:
        _embedded(session, key=f"{_PREFIX}/api-ingested/aa.npy")

    await run_sync_seed(seed)
    artifacts = Memory()
    await artifacts.put_bytes(
        storage.Object(f"{_PREFIX}/api-ingested/aa.npy"), b"v", content_type="x"
    )

    first = await ops.backfill_text_presence(index_db, artifacts, model=_MODEL)
    second = await ops.backfill_text_presence(index_db, artifacts, model=_MODEL)

    assert (first.marked_present, first.marked_absent) == (0, 1)
    assert (second.marked_present, second.marked_absent) == (0, 0)
    assert await _presence(index_db) == {f"{_PREFIX}/api-ingested/aa.npy": False}


async def test_backfill_batches_larger_than_one_page_of_keys(
    index_db: SavepointDb, run_sync_seed
) -> None:
    keys = [f"{_PREFIX}/api-ingested/{position:04d}.npy" for position in range(25)]

    def seed(session: Session) -> None:
        for key in keys:
            _embedded(session, key=key)

    await run_sync_seed(seed)
    artifacts = Memory()
    for key in keys:
        await artifacts.put_bytes(storage.Object(key), b"v", content_type="x")
        await artifacts.put_bytes(
            storage.Object(key.replace(".npy", "_text.npy")), b"t", content_type="x"
        )

    result = await ops.backfill_text_presence(index_db, artifacts, model=_MODEL, batch=4)

    assert result.marked_present == 25
    assert result.marked_absent == 0
    assert set((await _presence(index_db)).values()) == {True}
