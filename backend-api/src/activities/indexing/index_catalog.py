from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from shared.models import IndexBuild
from sqlalchemy.orm import Session


class StoredIndexArtifacts(Protocol):
    def key(self, version: str, filename: str) -> str: ...

    def list_stored_version_objects(self, version: str) -> list[tuple[str, int]]: ...

    def delete_stored_version(self, version: str) -> None: ...

    def delete_cached_version(self, version: str) -> None: ...


class ActiveIndexCatalog:
    def active_build(self, db: Session) -> IndexBuild | None:
        return db.query(IndexBuild).filter(IndexBuild.is_active).first()

    def list_builds_newest_first(self, db: Session) -> Sequence[IndexBuild]:
        return db.query(IndexBuild).order_by(IndexBuild.created_at.desc()).all()

    def add_inactive_build(
        self,
        db: Session,
        *,
        version: str,
        s3_key: str,
        embed_model: str,
        index_type: str,
        num_vectors: int,
        dimension: int,
    ) -> None:
        db.add(
            IndexBuild(
                version=version,
                s3_key=s3_key,
                embed_model=embed_model,
                index_type=index_type,
                num_vectors=num_vectors,
                dimension=dimension,
                is_active=False,
            )
        )
        db.commit()

    def mark_latest_available(
        self,
        db: Session,
        *,
        version: str,
        metadata: dict[str, Any],
        artifacts: StoredIndexArtifacts,
    ) -> None:
        db.query(IndexBuild).filter(
            IndexBuild.is_active,
            IndexBuild.version != version,
        ).update({"is_active": False})

        build = db.query(IndexBuild).filter(IndexBuild.version == version).first()
        if build is None:
            db.add(
                IndexBuild(
                    version=version,
                    s3_key=artifacts.key(version, "index.faiss"),
                    embed_model=metadata.get("model_name"),
                    index_type=metadata.get("index_type"),
                    num_vectors=metadata.get("num_vectors"),
                    dimension=metadata.get("dimension"),
                    is_active=True,
                )
            )
        else:
            build.is_active = True
            if not build.s3_key:
                build.s3_key = artifacts.key(version, "index.faiss")
            if build.embed_model is None:
                build.embed_model = metadata.get("model_name")
            if build.index_type is None:
                build.index_type = metadata.get("index_type")
            if build.num_vectors is None:
                build.num_vectors = metadata.get("num_vectors")
            if build.dimension is None:
                build.dimension = metadata.get("dimension")

        db.commit()

    def swap_to_version(self, version: str, db: Session) -> None:
        db.query(IndexBuild).filter(IndexBuild.is_active).update({"is_active": False})
        db.query(IndexBuild).filter(IndexBuild.version == version).update({"is_active": True})
        db.commit()

    def garbage_collect(
        self,
        db: Session,
        *,
        artifacts: StoredIndexArtifacts,
        retain_versions: int,
    ) -> list[str]:
        removed: list[str] = []

        for build in self.list_builds_newest_first(db)[retain_versions:]:
            if build.is_active:
                continue

            if build.s3_key:
                artifacts.delete_stored_version(build.version)

            artifacts.delete_cached_version(build.version)
            db.delete(build)
            removed.append(build.version)

        db.commit()
        return removed
