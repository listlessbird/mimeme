from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, Row, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mimeme import inference
from mimeme.db.schema import EmbeddingShard, IndexBuild, Processing, ProcessingStatus
from mimeme.index.model import Embedding, Manifest, Snapshot
from mimeme.job.model import ClaimOwnership
from mimeme.job.store import Store as JobStore


def _embedding(row: Row) -> Embedding:
    if row.embed_shard is not None and row.embed_row is not None and row.seq is not None:
        return Embedding(
            image_id=row.image_id,
            shard=row.embed_shard,
            row=row.embed_row,
            seq=row.seq,
            text_present=bool(row.embed_text_present),
        )
    image_key = str(row.embed_s3_key)
    return Embedding(
        image_id=row.image_id,
        image_key=image_key,
        text_key=inference.text_embedding_key(image_key) if row.embed_text_present else None,
    )


class Store:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def active_version(self) -> str | None:
        return await self._session.scalar(
            select(IndexBuild.version).where(IndexBuild.is_active.is_(True))
        )

    async def snapshot(self, *, model: str, target_generation: int) -> Snapshot:
        rows = (
            await self._session.execute(
                select(
                    Processing.image_id,
                    Processing.embed_s3_key,
                    Processing.embed_dim,
                    Processing.embed_text_present,
                    Processing.embed_shard,
                    Processing.embed_row,
                    EmbeddingShard.seq,
                )
                .outerjoin(
                    EmbeddingShard,
                    (EmbeddingShard.embed_model == Processing.embed_model)
                    & (EmbeddingShard.number == Processing.embed_shard),
                )
                .where(
                    Processing.embed_status == ProcessingStatus.DONE,
                    Processing.embed_model == model,
                    Processing.embed_s3_key.is_not(None),
                    Processing.embed_s3_key != "",
                )
                .order_by(Processing.image_id)
            )
        ).all()
        dimensions = {int(row.embed_dim) for row in rows if row.embed_dim is not None}
        if len(dimensions) > 1:
            raise ValueError(f"snapshot contains mixed embedding dimensions: {dimensions}")
        if rows and not dimensions:
            raise ValueError("snapshot embeddings have no recorded dimension")
        dimension = next(iter(dimensions), 0)
        if not rows:
            active = (
                await self._session.scalars(
                    select(IndexBuild).where(IndexBuild.is_active.is_(True))
                )
            ).first()
            dimension = active.dimension or 0 if active is not None else 0
        return Snapshot(
            target_generation=target_generation,
            dimension=dimension,
            embeddings=[_embedding(row) for row in rows],
        )

    async def unsealed(self, *, model: str, limit: int | None = None) -> list[Row]:
        query = (
            select(
                Processing.image_id,
                Processing.embed_s3_key,
                Processing.embed_text_present,
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

    async def mark_text_present(self, *, model: str, image_keys: list[str]) -> int:
        if not image_keys:
            return 0
        result = await self._session.execute(
            update(Processing)
            .where(
                Processing.embed_model == model,
                Processing.embed_text_present.is_(None),
                Processing.embed_s3_key.in_(image_keys),
            )
            .values(embed_text_present=True)
        )
        return cast("CursorResult[Any]", result).rowcount

    async def mark_text_absent(self, *, model: str) -> int:
        result = await self._session.execute(
            update(Processing)
            .where(
                Processing.embed_status == ProcessingStatus.DONE,
                Processing.embed_model == model,
                Processing.embed_text_present.is_(None),
            )
            .values(embed_text_present=False)
        )
        return cast("CursorResult[Any]", result).rowcount

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
            text_num_vectors=manifest.text_count,
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

    async def removable(self, *, protect: set[str], retain: int) -> list[str]:
        rows = (
            await self._session.scalars(
                select(IndexBuild).order_by(IndexBuild.created_at.desc(), IndexBuild.id.desc())
            )
        ).all()
        kept = 0
        removed: list[str] = []
        for row in rows:
            if row.version in protect or row.is_active:
                continue
            if kept < retain:
                kept += 1
            else:
                removed.append(row.version)
        return removed

    async def forget(self, versions: list[str]) -> None:
        if versions:
            await self._session.execute(
                delete(IndexBuild).where(
                    IndexBuild.version.in_(versions), IndexBuild.is_active.is_(False)
                )
            )
