from __future__ import annotations

import asyncio
import hashlib
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from mimeme.db.schema import Image, IngestURL, Processing, ProcessingStatus
from mimeme.ingest.facts import Facts
from mimeme.ingest.store import Store
from tests.factories import (
    create_annotation,
    create_image,
    create_ingest_url,
    create_job,
    create_processing,
)
from tests.job.conftest import PoolDb, SavepointDb

ZERO_PHASH = "0000000000000000"
NEAR_PHASH = "000000000000000f"  # hamming distance 4 from ZERO
FAR_PHASH = "ffffffffffffffff"  # hamming distance 64 from ZERO


def _facts(sha: str, phash: str) -> Facts:
    return Facts(sha256=sha, phash=phash, width=16, height=16, format="png", mode="RGB")


class TestLookups:
    async def test_find_by_sha(self, db: SavepointDb, run_sync_seed) -> None:
        image_id = await run_sync_seed(lambda s: create_image(session=s, sha256="a" * 64).id)
        async with db.read_session() as session:
            store = Store(session)
            assert await store.find_by_sha("a" * 64) == image_id
            assert await store.find_by_sha("b" * 64) is None

    async def test_find_by_phash_near_and_far(self, db: SavepointDb, run_sync_seed) -> None:
        image_id = await run_sync_seed(
            lambda s: create_image(session=s, sha256="c" * 64, phash=ZERO_PHASH).id
        )
        async with db.read_session() as session:
            store = Store(session)
            assert await store.find_by_phash(NEAR_PHASH) == image_id
            assert await store.find_by_phash(FAR_PHASH) is None
            assert await store.find_by_phash(None) is None


class TestInsert:
    async def test_insert_canonical_creates_image_and_processing(self, db: SavepointDb) -> None:
        async with db.write_session() as session:
            image_id = await Store(session).insert_canonical(
                facts=_facts("d" * 64, ZERO_PHASH),
                dataset="memes",
                filename="cat.png",
                s3_key="images/memes/dd/dd/x.png",
                etag="etag",
                file_size=123,
            )
        async with db.read_session() as session:
            image = await session.get(Image, image_id)
            assert image is not None
            assert image.sha256 == "d" * 64
            proc = await session.scalar(select(Processing).where(Processing.image_id == image_id))
            assert proc is not None


class TestDuplicateView:
    async def test_needs_flags_and_creates_processing(self, db: SavepointDb, run_sync_seed) -> None:
        def seed(s: Session) -> int:
            return create_image(session=s, sha256="e" * 64).id

        image_id = await run_sync_seed(seed)
        async with db.write_session() as session:
            view = await Store(session).duplicate_view(image_id)
        assert view.needs_annotation and view.needs_embedding
        async with db.read_session() as session:
            proc = await session.scalar(select(Processing).where(Processing.image_id == image_id))
            assert proc is not None

    async def test_fully_processed_needs_nothing(self, db: SavepointDb, run_sync_seed) -> None:
        def seed(s: Session) -> int:
            image = create_image(session=s, sha256="f" * 64)
            create_annotation(session=s, image=image)
            create_processing(
                session=s,
                image=image,
                caption_status=ProcessingStatus.DONE,
                ocr_status=ProcessingStatus.DONE,
                embed_status=ProcessingStatus.DONE,
                embed_s3_key="k",
            )
            return image.id

        image_id = await run_sync_seed(seed)
        async with db.write_session() as session:
            view = await Store(session).duplicate_view(image_id)
        assert not view.needs_annotation and not view.needs_embedding


class TestCounts:
    async def test_sweep_and_count(self, db: SavepointDb, run_sync_seed) -> None:
        def seed(s: Session) -> str:
            job = create_job(session=s)
            create_ingest_url(session=s, job=job, status=ProcessingStatus.DONE)
            create_ingest_url(session=s, job=job, status=ProcessingStatus.FAILED)
            create_ingest_url(session=s, job=job, status=ProcessingStatus.PENDING)
            return job.id

        job_id = await run_sync_seed(seed)
        async with db.write_session() as session:
            processed, failed, duplicates = await Store(session).sweep_and_count(job_id, "gave up")
        # one DONE non-dup, one FAILED, one PENDING swept -> failed. no duplicates.
        assert (processed, failed, duplicates) == (1, 2, 0)
        async with db.read_session() as session:
            swept = (
                await session.scalars(
                    select(IngestURL).where(IngestURL.status == ProcessingStatus.FAILED)
                )
            ).all()
            assert any(u.error_message == "gave up" for u in swept)


class TestAdvisoryLock:
    async def test_concurrent_identical_sha_yields_one_winner(
        self, pool_db: PoolDb, run_sync_seed
    ) -> None:
        # Seed nothing; two racers try to insert the same sha under the lock.
        sha = hashlib.sha256(uuid.uuid4().bytes).hexdigest()

        async def racer() -> str:
            async with pool_db.write_session() as session:
                store = Store(session)
                await store.acquire_dedup_lock()
                if await store.find_by_sha(sha) is not None:
                    return "duplicate"
                await store.insert_canonical(
                    facts=_facts(sha, ZERO_PHASH),
                    dataset=None,
                    filename=None,
                    s3_key=f"images/api-ingested/11/11/{sha}.png",
                    etag=None,
                    file_size=1,
                )
                return "winner"

        results = await asyncio.gather(racer(), racer())
        assert sorted(results) == ["duplicate", "winner"]
        async with pool_db.read_session() as session:
            count = len((await session.scalars(select(Image).where(Image.sha256 == sha))).all())
            assert count == 1
            # cleanup (pool_db is a real DB, not rolled back)
            image = (await session.scalars(select(Image).where(Image.sha256 == sha))).first()
        async with pool_db.write_session() as session:
            obj = await session.get(Image, image.id)
            await session.delete(obj)
