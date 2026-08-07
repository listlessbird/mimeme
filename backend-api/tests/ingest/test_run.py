from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from mimeme import inference, storage
from mimeme.db.schema import (
    Annotation,
    DuplicateReason,
    Image,
    IngestURL,
    Processing,
    ProcessingStatus,
)
from mimeme.ingest import rule
from mimeme.ingest.model import Input, RemoteUrl, Retryable, Staged
from mimeme.ingest.run import run
from tests.factories import (
    create_annotation,
    create_image,
    create_ingest_url,
    create_job,
    create_processing,
    create_search_index_state,
)
from tests.ingest.conftest import FakeEnv, FakeImages, FakeInference, facts_for, image_http
from tests.job.conftest import SavepointDb

REMOTE_URL = "https://example.com/cat.png"


async def _seed_job_item(run_sync_seed) -> tuple[str, int]:
    def seed(s: Session) -> tuple[str, int]:
        create_search_index_state(session=s, desired_generation=0, active_generation=0)
        job = create_job(session=s)
        url = create_ingest_url(session=s, job=job)
        s.flush()
        return job.id, url.id

    return await run_sync_seed(seed)


def _remote_env(db: SavepointDb, data: bytes, facts_map: dict[str, object]) -> FakeEnv:
    images = FakeImages()
    for key, facts in facts_map.items():
        images.set(key, facts)
    return FakeEnv(db=db, http=image_http(data), image_facts=images)


class TestRemote:
    async def test_new_image_processed_end_to_end(
        self, db: SavepointDb, run_sync_seed, png_bytes: bytes
    ) -> None:
        job_id, item_id = await _seed_job_item(run_sync_seed)
        facts = facts_for(png_bytes, phash="0000000000000000")
        env = _remote_env(db, png_bytes, {rule.staging_key(item_id): facts})

        result = await run(
            env,
            Input(job_id=job_id, item_id=item_id, source=RemoteUrl(url=REMOTE_URL), dataset="d"),
        )

        assert result.outcome == "processed"
        canonical = rule.canonical_media_key(sha256=facts.sha256, dataset="d", image_format="png")
        assert await env.media.stat(storage.Object(canonical)) is not None
        # provisional staging object cleaned up
        assert await env.artifacts.stat(storage.Object(rule.staging_key(item_id))) is None
        assert len(env.inference.annotate_calls) == 1
        assert len(env.inference.embed_calls) == 1
        async with db.read_session() as session:
            image = await session.scalar(select(Image).where(Image.sha256 == facts.sha256))
            assert image is not None and image.s3_key == canonical
            url = await session.get(IngestURL, item_id)
            assert url.status is ProcessingStatus.DONE and url.image_id == image.id
            ann = await session.scalar(select(Annotation).where(Annotation.image_id == image.id))
            assert ann is not None and ann.caption_text == "a caption"

    async def test_non_image_content_type_fails_terminally(
        self, db: SavepointDb, run_sync_seed
    ) -> None:
        job_id, item_id = await _seed_job_item(run_sync_seed)
        env = FakeEnv(
            db=db,
            http=image_http(b"<html>", content_type="text/html"),
            image_facts=FakeImages(),
        )
        result = await run(
            env, Input(job_id=job_id, item_id=item_id, source=RemoteUrl(url=REMOTE_URL))
        )
        assert result.outcome == "failed"
        async with db.read_session() as session:
            url = await session.get(IngestURL, item_id)
            assert url.status is ProcessingStatus.FAILED

    async def test_retryable_inference_propagates_without_marking_done(
        self, db: SavepointDb, run_sync_seed, png_bytes: bytes
    ) -> None:
        job_id, item_id = await _seed_job_item(run_sync_seed)
        facts = facts_for(png_bytes, phash="0000000000000000")
        env = _remote_env(db, png_bytes, {rule.staging_key(item_id): facts})
        env.inference = FakeInference(fail_annotation=inference.Unavailable("gpu down"))

        with pytest.raises(Retryable):
            await run(env, Input(job_id=job_id, item_id=item_id, source=RemoteUrl(url=REMOTE_URL)))
        async with db.read_session() as session:
            url = await session.get(IngestURL, item_id)
            assert url.status is not ProcessingStatus.DONE
        # staging object retained for the retry
        assert await env.artifacts.stat(storage.Object(rule.staging_key(item_id))) is not None

    async def test_soft_embed_failure_still_processes(
        self, db: SavepointDb, run_sync_seed, png_bytes: bytes
    ) -> None:
        job_id, item_id = await _seed_job_item(run_sync_seed)
        facts = facts_for(png_bytes, phash="0000000000000000")
        env = _remote_env(db, png_bytes, {rule.staging_key(item_id): facts})
        env.inference = FakeInference(embed_ok=False)

        result = await run(
            env, Input(job_id=job_id, item_id=item_id, source=RemoteUrl(url=REMOTE_URL))
        )
        assert result.outcome == "processed"
        async with db.read_session() as session:
            image = await session.scalar(select(Image).where(Image.sha256 == facts.sha256))
            proc = await session.scalar(select(Processing).where(Processing.image_id == image.id))
            assert proc.embed_status is not ProcessingStatus.DONE


class TestStaged:
    async def test_new_staged_upload_processed_and_cleaned(
        self, db: SavepointDb, run_sync_seed, png_bytes: bytes
    ) -> None:
        job_id, item_id = await _seed_job_item(run_sync_seed)
        artifact_key = "uploads/staging/abc.png"
        env = FakeEnv(db=db, image_facts=FakeImages())
        await env.artifacts.put_bytes(
            storage.Object(artifact_key), png_bytes, content_type="image/png"
        )
        facts = facts_for(png_bytes, phash="0000000000000000")
        env.image_facts.set(artifact_key, facts)

        result = await run(
            env,
            Input(job_id=job_id, item_id=item_id, source=Staged(artifact_key=artifact_key)),
        )
        assert result.outcome == "processed"
        canonical = rule.canonical_media_key(sha256=facts.sha256, dataset=None, image_format="png")
        assert await env.media.stat(storage.Object(canonical)) is not None
        # staged object deleted under the success rule
        assert await env.artifacts.stat(storage.Object(artifact_key)) is None


class TestDuplicates:
    async def test_exact_sha_duplicate(
        self, db: SavepointDb, run_sync_seed, png_bytes: bytes
    ) -> None:
        facts = facts_for(png_bytes, phash="0000000000000000")

        def seed(s: Session) -> tuple[str, int]:
            create_search_index_state(session=s, desired_generation=0, active_generation=0)
            job = create_job(session=s)
            url = create_ingest_url(session=s, job=job)
            existing = create_image(session=s, sha256=facts.sha256, phash="ffffffffffffffff")
            create_annotation(session=s, image=existing)
            create_processing(
                session=s,
                image=existing,
                caption_status=ProcessingStatus.DONE,
                ocr_status=ProcessingStatus.DONE,
                embed_status=ProcessingStatus.DONE,
                embed_s3_key="k",
            )
            s.flush()
            return job.id, url.id

        job_id, item_id = await run_sync_seed(seed)
        env = _remote_env(db, png_bytes, {rule.staging_key(item_id): facts})

        result = await run(
            env, Input(job_id=job_id, item_id=item_id, source=RemoteUrl(url=REMOTE_URL))
        )
        assert result.outcome == "duplicate"
        assert result.duplicate_reason is DuplicateReason.SHA256
        # fully-processed original needs no re-inference
        assert env.inference.annotate_calls == []
        async with db.read_session() as session:
            count = len(
                (await session.scalars(select(Image).where(Image.sha256 == facts.sha256))).all()
            )
            assert count == 1
            url = await session.get(IngestURL, item_id)
            assert url.status is ProcessingStatus.DONE
            assert url.duplicate_reason is DuplicateReason.SHA256

    async def test_phash_duplicate(self, db: SavepointDb, run_sync_seed, png_bytes: bytes) -> None:
        facts = facts_for(png_bytes, phash="000000000000000f")  # near 0

        def seed(s: Session) -> tuple[str, int]:
            create_search_index_state(session=s, desired_generation=0, active_generation=0)
            job = create_job(session=s)
            url = create_ingest_url(session=s, job=job)
            create_image(session=s, sha256="9" * 64, phash="0000000000000000")
            s.flush()
            return job.id, url.id

        job_id, item_id = await run_sync_seed(seed)
        env = _remote_env(db, png_bytes, {rule.staging_key(item_id): facts})

        result = await run(
            env, Input(job_id=job_id, item_id=item_id, source=RemoteUrl(url=REMOTE_URL))
        )
        assert result.outcome == "duplicate"
        assert result.duplicate_reason is DuplicateReason.PHASH
        async with db.read_session() as session:
            # no new canonical image inserted for the near-duplicate sha
            assert await session.scalar(select(Image).where(Image.sha256 == facts.sha256)) is None
