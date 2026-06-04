from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Protocol

import structlog

log = structlog.get_logger()


class IndexStorage(Protocol):
    INDEXES_PREFIX: ClassVar[str]

    def build_index_key(self, version: str, filename: str) -> str: ...

    def exists(self, key: str) -> bool: ...

    def list_objects(self, prefix: str) -> list[tuple[str, int]]: ...

    def download_file(self, key: str, local_path: Path) -> None: ...

    def upload_file(self, local_path: Path, key: str, content_type: str | None = None) -> str: ...

    def delete(self, key: str) -> None: ...


class IndexArtifactPaths:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.index_file = cache_path / "index.faiss"
        self.mapping_file = cache_path / "mapping.json"
        self.metadata_file = cache_path / "metadata.json"
        self.text_index_file = cache_path / "text_index.faiss"
        self.text_mapping_file = cache_path / "text_mapping.json"
        self.text_metadata_file = cache_path / "text_metadata.json"


class IndexArtifactStore:
    def __init__(self, *, storage: IndexStorage, cache_dir: Path) -> None:
        self._storage = storage
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def version_sort_key(version: str) -> tuple[int, str]:
        match = re.match(r"^v(\d{8})-(\d{6})(?:-(.*))?$", version)
        if not match:
            return (0, version)
        stamp = int(f"{match.group(1)}{match.group(2)}")
        suffix = match.group(3) or ""
        return (stamp, suffix)

    def cache_path(self, version: str) -> Path:
        return self._cache_dir / version

    def paths(self, version: str) -> IndexArtifactPaths:
        return IndexArtifactPaths(self.cache_path(version))

    def key(self, version: str, filename: str) -> str:
        return self._storage.build_index_key(version, filename)

    def list_versions(self) -> list[str]:
        prefix = f"{self._storage.INDEXES_PREFIX}/"
        versions: set[str] = set()
        for key, _ in self._storage.list_objects(prefix):
            parts = key.split("/", 2)
            if len(parts) >= 2 and parts[1]:
                versions.add(parts[1])
        return sorted(versions, key=self.version_sort_key)

    def has_required_artifacts(self, version: str) -> bool:
        return self._storage.exists(self.key(version, "index.faiss")) and self._storage.exists(
            self.key(version, "mapping.json")
        )

    def ensure_cached(self, version: str) -> IndexArtifactPaths:
        paths = self.paths(version)
        self._ensure_image_artifacts(version, paths)
        self._ensure_text_artifacts(version, paths)
        return paths

    def has_text_artifacts_cached(self, version: str) -> bool:
        paths = self.paths(version)
        return paths.text_index_file.exists()

    def read_mapping(self, path: Path) -> dict[int, int]:
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as file:
            raw = json.load(file)
        return {int(key): value for key, value in raw.items()}

    def read_metadata(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    def write_json(self, path: Path, value: object, *, indent: int | None = None) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(value, file, indent=indent)

    def upload_files(self, version: str, local_paths: list[Path]) -> list[tuple[Path, str]]:
        upload_jobs = [(path, self.key(version, path.name)) for path in local_paths]
        try:
            with ThreadPoolExecutor(max_workers=len(upload_jobs)) as executor:
                futures = [
                    executor.submit(self._storage.upload_file, local_path, key)
                    for local_path, key in upload_jobs
                ]
                for future in futures:
                    future.result()
        except Exception:
            self.cleanup_stored_version(version)
            raise
        return upload_jobs

    def copy_to_cache(self, version: str, local_paths: list[Path]) -> None:
        cache_path = self.cache_path(version)
        cache_path.mkdir(parents=True, exist_ok=True)
        for local_path in local_paths:
            shutil.copy(local_path, cache_path / local_path.name)

    def cleanup_stored_version(self, version: str) -> None:
        prefix = f"{self._storage.INDEXES_PREFIX}/{version}"
        for key, _ in self._storage.list_objects(prefix):
            try:
                self._storage.delete(key)
            except Exception:
                pass

    def delete_stored_version(self, version: str) -> None:
        objects = self.list_stored_version_objects(version)
        if not objects:
            return
        with ThreadPoolExecutor(max_workers=min(8, len(objects))) as executor:
            futures = [executor.submit(self._storage.delete, key) for key, _ in objects]
            for future in futures:
                future.result()

    def delete_cached_version(self, version: str) -> None:
        cache_path = self.cache_path(version)
        if cache_path.exists():
            shutil.rmtree(cache_path)

    def list_stored_version_objects(self, version: str) -> list[tuple[str, int]]:
        prefix = f"{self._storage.INDEXES_PREFIX}/{version}"
        return self._storage.list_objects(prefix)

    def _ensure_image_artifacts(self, version: str, paths: IndexArtifactPaths) -> None:
        if paths.index_file.exists():
            return

        paths.cache_path.mkdir(parents=True, exist_ok=True)
        download_jobs: list[tuple[str, Path]] = [
            (self.key(version, "index.faiss"), paths.index_file),
            (self.key(version, "mapping.json"), paths.mapping_file),
        ]
        metadata_key = self.key(version, "metadata.json")
        if self._storage.exists(metadata_key):
            download_jobs.append((metadata_key, paths.metadata_file))
        self._download(download_jobs)

    def _ensure_text_artifacts(self, version: str, paths: IndexArtifactPaths) -> None:
        if paths.text_index_file.exists() and paths.text_mapping_file.exists():
            return
        try:
            text_index_key = self.key(version, "text_index.faiss")
            if not self._storage.exists(text_index_key):
                return

            paths.cache_path.mkdir(parents=True, exist_ok=True)
            download_jobs: list[tuple[str, Path]] = [
                (text_index_key, paths.text_index_file),
                (self.key(version, "text_mapping.json"), paths.text_mapping_file),
            ]
            text_metadata_key = self.key(version, "text_metadata.json")
            if self._storage.exists(text_metadata_key):
                download_jobs.append((text_metadata_key, paths.text_metadata_file))
            self._download(download_jobs)
        except Exception:
            log.warning("text_index_fetch_failed", version=version, exc_info=True)

    def _download(self, jobs: list[tuple[str, Path]]) -> None:
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [
                executor.submit(self._storage.download_file, key, path) for key, path in jobs
            ]
            for future in futures:
                future.result()
