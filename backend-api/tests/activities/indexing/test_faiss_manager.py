from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from activities.indexing import faiss_manager as faiss_manager_module
from shared.models import IndexBuild


class _FakeFaissIndex:
    def __init__(self, ntotal: int = 1, d: int = 3, score: float = 0.75, idx: int = 0) -> None:
        self.ntotal = ntotal
        self.d = d
        self._score = score
        self._idx = idx

    def search(self, _query_vector, _k: int = 20):
        distances = np.array([[self._score]], dtype=np.float32)
        indices = np.array([[self._idx]], dtype=np.int64)
        return distances, indices


class _FakeStorage:
    INDEXES_PREFIX = "indexes"

    def __init__(
        self,
        *,
        objects: list[str] | None = None,
        existing_keys: set[str] | None = None,
        fail_suffixes: set[str] | None = None,
        fail_upload_suffixes: set[str] | None = None,
    ) -> None:
        self.objects = objects or []
        self.existing_keys = existing_keys or set()
        self.fail_suffixes = fail_suffixes or set()
        self.fail_upload_suffixes = fail_upload_suffixes or set()
        self.download_calls: list[str] = []
        self.upload_calls: list[str] = []
        self.delete_calls: list[str] = []

    def build_index_key(self, version: str, filename: str) -> str:
        return f"{self.INDEXES_PREFIX}/{version}/{filename}"

    def exists(self, key: str) -> bool:
        return key in self.existing_keys

    def download_file(self, key: str, path: Path) -> None:
        self.download_calls.append(key)
        if any(key.endswith(suffix) for suffix in self.fail_suffixes):
            raise RuntimeError(f"simulated download failure for {key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if key.endswith("mapping.json") and not key.endswith("text_mapping.json"):
            path.write_text(json.dumps({"0": 111}), encoding="utf-8")
            return
        if key.endswith("text_mapping.json"):
            path.write_text(json.dumps({"0": 222}), encoding="utf-8")
            return
        if key.endswith("metadata.json") and not key.endswith("text_metadata.json"):
            path.write_text(
                json.dumps({"model_name": "siglip", "index_type": "flat", "num_vectors": 1, "dimension": 3}),
                encoding="utf-8",
            )
            return
        if key.endswith("text_metadata.json"):
            path.write_text(json.dumps({"kind": "text"}), encoding="utf-8")
            return
        path.write_bytes(b"fake-index")

    def list_objects(self, _prefix: str):
        return [(key, 1) for key in self.objects if key.startswith(_prefix)]

    def upload_file(self, local_path: Path, key: str) -> None:
        self.upload_calls.append(key)
        if any(key.endswith(suffix) for suffix in self.fail_upload_suffixes):
            raise RuntimeError(f"simulated upload failure for {key}")
        if key not in self.objects:
            self.objects.append(key)
        self.existing_keys.add(key)

    def delete(self, key: str) -> None:
        self.delete_calls.append(key)
        if key in self.objects:
            self.objects.remove(key)
        if key in self.existing_keys:
            self.existing_keys.remove(key)


class _LogCapture:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []

    def warning(self, event: str, **kwargs) -> None:
        self.warnings.append((event, kwargs))


def _new_manager(tmp_path, monkeypatch, storage: _FakeStorage) -> faiss_manager_module.FaissIndexManager:
    monkeypatch.setattr(faiss_manager_module.settings, "index_cache_dir", tmp_path)
    monkeypatch.setattr(faiss_manager_module, "get_storage_service", lambda: storage)
    return faiss_manager_module.FaissIndexManager()


def _seed_cached_index_version(
    base_dir: Path,
    version: str,
    *,
    mapping: dict[str, int] | None = None,
    metadata: dict | None = None,
) -> None:
    cache_path = base_dir / version
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "index.faiss").write_bytes(b"dummy")
    (cache_path / "mapping.json").write_text(json.dumps(mapping or {"0": 123}), encoding="utf-8")
    if metadata is not None:
        (cache_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_search_raises_when_image_index_not_loaded(tmp_path, monkeypatch) -> None:
    manager = _new_manager(tmp_path, monkeypatch, _FakeStorage())
    with pytest.raises(ValueError, match="No index is loaded"):
        manager.search(np.array([0.1, 0.2, 0.3], dtype=np.float32), k=5)


def test_load_active_index_raises_when_no_active_build(session_factory, tmp_path, monkeypatch) -> None:
    manager = _new_manager(tmp_path, monkeypatch, _FakeStorage())
    with session_factory() as db:
        with pytest.raises(FileNotFoundError, match="No active index found in db"):
            manager.load_active_index(db)


def test_load_active_index_loads_current_active_version(session_factory, tmp_path, monkeypatch) -> None:
    manager = _new_manager(tmp_path, monkeypatch, _FakeStorage())
    version = "v20260101-000000-aaaa1111"
    _seed_cached_index_version(tmp_path, version)
    monkeypatch.setattr(faiss_manager_module.faiss, "read_index", lambda _path: _FakeFaissIndex(ntotal=1, d=3))

    with session_factory() as db:
        db.add(IndexBuild(version=version, is_active=True))
        db.commit()
        manager.load_active_index(db)

    assert manager.is_loaded is True
    assert manager.active_version == version
    assert manager.num_vectors == 1


def test_autoload_latest_available_activates_latest_valid_version(
    session_factory, tmp_path, monkeypatch
) -> None:
    v1 = "v20260101-000000-aaaa1111"
    v2 = "v20260102-000000-bbbb2222"
    invalid = "v20260103-000000-cccc3333"

    storage = _FakeStorage(
        objects=[
            f"indexes/{v1}/index.faiss",
            f"indexes/{v2}/index.faiss",
            f"indexes/{invalid}/index.faiss",
        ],
        existing_keys={
            f"indexes/{v1}/index.faiss",
            f"indexes/{v1}/mapping.json",
            f"indexes/{v2}/index.faiss",
            f"indexes/{v2}/mapping.json",
            f"indexes/{v2}/metadata.json",
            # invalid version missing mapping.json on purpose
            f"indexes/{invalid}/index.faiss",
        },
    )
    manager = _new_manager(tmp_path, monkeypatch, storage)
    monkeypatch.setattr(faiss_manager_module.faiss, "read_index", lambda _path: _FakeFaissIndex())

    with session_factory() as db:
        db.add(IndexBuild(version=v1, is_active=True))
        db.commit()

        loaded_version = manager.autoload_latest_available(db)
        assert loaded_version == v2

        rows = db.query(IndexBuild).order_by(IndexBuild.version).all()
        by_version = {row.version: row for row in rows}
        assert by_version[v1].is_active is False
        assert by_version[v2].is_active is True
        assert by_version[v2].index_type == "flat"
        assert by_version[v2].num_vectors == 1
        assert by_version[v2].dimension == 3


def test_load_index_version_continues_when_text_index_fetch_fails(tmp_path, monkeypatch) -> None:
    version = "v20260101-000000-test"
    _seed_cached_index_version(tmp_path, version)

    text_index_key = f"indexes/{version}/text_index.faiss"
    text_mapping_key = f"indexes/{version}/text_mapping.json"
    fake_storage = _FakeStorage(
        existing_keys={text_index_key, text_mapping_key},
        fail_suffixes={"text_index.faiss"},
    )
    log_capture = _LogCapture()

    manager = _new_manager(tmp_path, monkeypatch, fake_storage)
    monkeypatch.setattr(faiss_manager_module.structlog, "get_logger", lambda: log_capture)
    monkeypatch.setattr(faiss_manager_module.faiss, "read_index", lambda _path: _FakeFaissIndex())

    manager.load_index_version(version)

    assert manager.is_loaded is True
    assert manager.active_version == version
    assert manager.num_vectors == 1
    assert manager.has_text_index() is False
    assert len(fake_storage.download_calls) >= 1
    assert len(log_capture.warnings) == 1
    event, fields = log_capture.warnings[0]
    assert event == "text_index_fetch_failed"
    assert fields["version"] == version


def test_search_text_raises_when_no_active_version(tmp_path, monkeypatch) -> None:
    manager = _new_manager(tmp_path, monkeypatch, _FakeStorage())
    with pytest.raises(ValueError, match="No active index version"):
        manager.search_text(np.array([0.1, 0.2, 0.3], dtype=np.float32), k=5)


def test_search_text_lazily_loads_text_index_and_maps_results(tmp_path, monkeypatch) -> None:
    version = "v20260105-000000-textok"
    cache_path = tmp_path / version
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "text_index.faiss").write_bytes(b"fake")
    (cache_path / "text_mapping.json").write_text(json.dumps({"0": 9001}), encoding="utf-8")

    manager = _new_manager(tmp_path, monkeypatch, _FakeStorage())
    manager._active_version = version
    monkeypatch.setattr(faiss_manager_module.faiss, "read_index", lambda _path: _FakeFaissIndex(score=0.91, idx=0))

    results = manager.search_text(np.array([0.1, 0.2, 0.3], dtype=np.float32), k=5)

    assert manager.is_text_loaded is True
    assert manager.has_text_index() is True
    assert len(results) == 1
    assert results[0][0] == 9001
    assert results[0][1] == pytest.approx(0.91, rel=1e-6)


def test_build_index_raises_on_mismatched_input_lengths(tmp_path, monkeypatch) -> None:
    manager = _new_manager(tmp_path, monkeypatch, _FakeStorage())
    with pytest.raises(ValueError, match="Embeddings and image_ids must have same length"):
        manager.build_index(
            embeddings=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            image_ids=[1, 2],
            model_name="siglip",
        )


def test_build_index_upload_failure_cleans_up_uploaded_prefix(tmp_path, monkeypatch) -> None:
    storage = _FakeStorage(fail_upload_suffixes={"mapping.json"})
    manager = _new_manager(tmp_path, monkeypatch, storage)

    monkeypatch.setattr(faiss_manager_module.faiss, "normalize_L2", lambda _arr: None)
    monkeypatch.setattr(
        manager,
        "_create_faiss_index",
        lambda *_args, **_kwargs: _FakeFaissIndex(ntotal=1, d=3),
    )
    monkeypatch.setattr(faiss_manager_module.faiss, "write_index", lambda _idx, path: Path(path).write_bytes(b"x"))

    with pytest.raises(RuntimeError, match="simulated upload failure"):
        manager.build_index(
            embeddings=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            image_ids=[1],
            model_name="siglip",
            index_type="flat",
        )

    # Outcome-focused: partial uploads are cleaned up from storage prefix.
    assert storage.objects == []


def test_build_index_creates_db_record_when_db_provided(session_factory, tmp_path, monkeypatch) -> None:
    storage = _FakeStorage()
    manager = _new_manager(tmp_path, monkeypatch, storage)

    monkeypatch.setattr(faiss_manager_module.faiss, "normalize_L2", lambda _arr: None)
    monkeypatch.setattr(
        manager,
        "_create_faiss_index",
        lambda *_args, **_kwargs: _FakeFaissIndex(ntotal=2, d=3),
    )
    monkeypatch.setattr(faiss_manager_module.faiss, "write_index", lambda _idx, path: Path(path).write_bytes(b"x"))

    with session_factory() as db:
        result = manager.build_index(
            embeddings=np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=np.float32),
            image_ids=[10, 11],
            model_name="siglip",
            index_type="flat",
            db=db,
        )
        row = db.query(IndexBuild).filter(IndexBuild.version == result.version).first()
        assert row is not None
        assert row.is_active is False
        assert row.num_vectors == 2
        assert row.dimension == 3
        assert row.index_type == "flat"
        assert row.embed_model == "siglip"


def test_swap_to_version_activates_target_version(session_factory, tmp_path, monkeypatch) -> None:
    manager = _new_manager(tmp_path, monkeypatch, _FakeStorage())
    old_version = "v-old"
    new_version = "v-new"
    _seed_cached_index_version(tmp_path, old_version, mapping={"0": 1})
    _seed_cached_index_version(tmp_path, new_version, mapping={"0": 2})
    monkeypatch.setattr(faiss_manager_module.faiss, "read_index", lambda _path: _FakeFaissIndex(ntotal=1, d=3))

    with session_factory() as db:
        old_v = IndexBuild(version=old_version, is_active=True)
        new_v = IndexBuild(version=new_version, is_active=False)
        db.add_all([old_v, new_v])
        db.commit()

        manager.swap_to_version(new_version, db)

        rows = db.query(IndexBuild).order_by(IndexBuild.version).all()
        by_version = {row.version: row for row in rows}
        assert by_version[old_version].is_active is False
        assert by_version[new_version].is_active is True
        assert manager.active_version == new_version


def test_garbage_collect_removes_old_inactive_versions_and_keeps_active(
    session_factory, tmp_path, monkeypatch
) -> None:
    now = datetime.now(UTC)
    keep_latest = "v-keep-latest"
    active_old = "v-active-old"
    stale_old = "v-stale-old"

    storage = _FakeStorage(
        objects=[
            f"indexes/{active_old}/index.faiss",
            f"indexes/{stale_old}/index.faiss",
            f"indexes/{stale_old}/mapping.json",
        ]
    )
    manager = _new_manager(tmp_path, monkeypatch, storage)

    # Prepare cache dirs for both removable/kept versions.
    (tmp_path / keep_latest).mkdir(parents=True, exist_ok=True)
    (tmp_path / active_old).mkdir(parents=True, exist_ok=True)
    (tmp_path / stale_old).mkdir(parents=True, exist_ok=True)

    with session_factory() as db:
        db.add_all(
            [
                IndexBuild(version=keep_latest, is_active=False, created_at=now),
                IndexBuild(
                    version=active_old,
                    is_active=True,
                    s3_key=f"indexes/{active_old}/index.faiss",
                    created_at=now - timedelta(minutes=1),
                ),
                IndexBuild(
                    version=stale_old,
                    is_active=False,
                    s3_key=f"indexes/{stale_old}/index.faiss",
                    created_at=now - timedelta(minutes=2),
                ),
            ]
        )
        db.commit()

        removed = manager.garbage_collect(db, retain_versions=1)

        assert removed == [stale_old]
        remaining_versions = {row.version for row in db.query(IndexBuild).all()}
        assert keep_latest in remaining_versions
        assert active_old in remaining_versions
        assert stale_old not in remaining_versions

    assert (tmp_path / stale_old).exists() is False
    assert (tmp_path / active_old).exists() is True
    assert not any(key.startswith(f"indexes/{stale_old}/") for key in storage.objects)
    assert any(key.startswith(f"indexes/{active_old}/") for key in storage.objects)
