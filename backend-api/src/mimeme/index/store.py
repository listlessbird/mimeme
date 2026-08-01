from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mimeme.db.schema import IndexBuild, Processing, ProcessingStatus
from mimeme.index.model import Embedding, Manifest, Snapshot
from mimeme.job.store import Store as JobStore


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
                select(Processing.image_id, Processing.embed_s3_key, Processing.embed_dim)
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
            embeddings=[
                Embedding(image_id=row.image_id, image_key=str(row.embed_s3_key)) for row in rows
            ],
        )

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
