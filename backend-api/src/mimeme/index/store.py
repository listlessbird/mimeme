from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Row, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mimeme.db.schema import EmbeddingShard, IndexBuild, Processing, ProcessingStatus
from mimeme.index import documents
from mimeme.index.model import Manifest, Snapshot
from mimeme.job.model import ClaimOwnership
from mimeme.job.store import Store as JobStore


class Store:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def active_version(self) -> str | None:
        return await self._session.scalar(
            select(IndexBuild.version).where(IndexBuild.is_active.is_(True))
        )

    async def deactivate(self, version: str) -> None:
        await self._session.execute(
            update(IndexBuild)
            .where(IndexBuild.version == version, IndexBuild.is_active.is_(True))
            .values(is_active=False)
        )

    async def snapshot(self, *, model: str, target_generation: int) -> Snapshot:
        snapshot = await documents.capture(
            self._session,
            model=model,
            target_generation=target_generation,
        )
        if not snapshot.embeddings:
            active = (
                await self._session.scalars(
                    select(IndexBuild).where(IndexBuild.is_active.is_(True))
                )
            ).first()
            dimension = active.dimension or 0 if active is not None else 0
            return snapshot.model_copy(update={"dimension": dimension})
        return snapshot

    async def unsealed(self, *, model: str, limit: int | None = None) -> list[Row]:
        query = (
            select(
                Processing.image_id,
                Processing.embed_s3_key,
            )
            .where(
                Processing.embed_status == ProcessingStatus.DONE,
                Processing.embed_model == model,
                Processing.embed_s3_key.is_not(None),
                Processing.embed_s3_key != "",
                Processing.embed_shard.is_(None),
            )
            .order_by(Processing.image_id)
        )
        if limit is not None:
            query = query.limit(limit)
        return list((await self._session.execute(query)).all())

    async def next_shard(self, *, model: str) -> int:
        highest = await self._session.scalar(
            select(func.max(EmbeddingShard.number)).where(EmbeddingShard.embed_model == model)
        )
        return 0 if highest is None else int(highest) + 1

    async def open_shard(self, *, model: str) -> EmbeddingShard | None:
        return (
            await self._session.scalars(
                select(EmbeddingShard).where(
                    EmbeddingShard.embed_model == model,
                    EmbeddingShard.sealed.is_(False),
                )
            )
        ).first()

    async def record_shard(
        self,
        *,
        model: str,
        shard: int,
        seq: int,
        first_row: int,
        image_ids: list[int],
        sealed: bool,
    ) -> None:
        for offset, image_id in enumerate(image_ids):
            await self._session.execute(
                update(Processing)
                .where(Processing.image_id == image_id, Processing.embed_shard.is_(None))
                .values(embed_shard=shard, embed_row=first_row + offset)
            )
        row = (
            await self._session.scalars(
                select(EmbeddingShard).where(
                    EmbeddingShard.embed_model == model, EmbeddingShard.number == shard
                )
            )
        ).first()
        if row is None:
            row = EmbeddingShard(embed_model=model, number=shard)
            self._session.add(row)
        row.seq = seq
        row.row_count = first_row + len(image_ids)
        row.sealed = sealed
        await self._session.flush()

    async def activate(self, *, job_id: str, manifest: Manifest) -> None:
        await self._session.execute(
            update(IndexBuild).where(IndexBuild.is_active.is_(True)).values(is_active=False)
        )
        row = (
            await self._session.scalars(
                select(IndexBuild).where(IndexBuild.version == manifest.version)
            )
        ).first()
        if row is None:
            row = IndexBuild(version=manifest.version)
            self._session.add(row)
        row.s3_key = next(file.key for file in manifest.files if file.name == "index.faiss")
        row.embed_model = manifest.model
        row.index_type = manifest.index_type
        row.num_vectors = manifest.image_count
        row.dimension = manifest.dimension
        row.is_active = True
        jobs = JobStore(self._session)
        await jobs.activate(
            job_id=job_id,
            target_generation=manifest.target_generation,
            reconciled_at=datetime.now(UTC),
        )
        await jobs.complete_rebuild(
            job_id=job_id,
            version=manifest.version,
            num_vectors=manifest.image_count,
            dimension=manifest.dimension,
            removed_versions=[],
            text_num_vectors=None,
        )
        await jobs.release(job_id=job_id)

    async def reconcile_empty(self, *, job_id: str, target_generation: int) -> None:
        jobs = JobStore(self._session)
        await jobs.activate(
            job_id=job_id,
            target_generation=target_generation,
            reconciled_at=datetime.now(UTC),
        )
        await jobs.complete_rebuild(
            job_id=job_id,
            version="",
            num_vectors=0,
            dimension=0,
            removed_versions=[],
            text_num_vectors=None,
        )
        await jobs.release(job_id=job_id)

    async def fail(self, *, job_id: str, error: str, cancelled: bool) -> None:
        jobs = JobStore(self._session)
        claim = (await jobs.freshness()).active_claim
        if claim is None or claim.job_id != job_id:
            raise ClaimOwnership(f"Job {job_id} does not own the rebuild claim")
        if cancelled:
            await jobs.set_cancelled(job_id)
        else:
            await jobs.fail_rebuild(job_id, error)
        await jobs.release(job_id=job_id)
