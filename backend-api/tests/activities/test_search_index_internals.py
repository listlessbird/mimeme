from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


def test_artifact_store_fetches_text_artifacts_when_image_artifacts_are_cached(
    tmp_path: Path,
) -> None:
    from activities.indexing.index_artifacts import IndexArtifactStore

    storage = MagicMock()
    storage.INDEXES_PREFIX = "indexes"
    storage.build_index_key.side_effect = lambda version, filename: f"indexes/{version}/{filename}"
    storage.exists.side_effect = lambda key: (
        key
        in {
            "indexes/v1/text_index.faiss",
            "indexes/v1/text_mapping.json",
            "indexes/v1/text_metadata.json",
        }
    )

    cached = tmp_path / "v1"
    cached.mkdir()
    (cached / "index.faiss").write_bytes(b"cached")
    (cached / "mapping.json").write_text("{}", encoding="utf-8")

    store = IndexArtifactStore(storage=storage, cache_dir=tmp_path)

    store.ensure_cached("v1")

    downloaded = {call.args[0] for call in storage.download_file.call_args_list}
    assert downloaded == {
        "indexes/v1/text_index.faiss",
        "indexes/v1/text_mapping.json",
        "indexes/v1/text_metadata.json",
    }


def test_artifact_store_lists_versions_by_timestamp_and_requires_image_artifacts(
    tmp_path: Path,
) -> None:
    from activities.indexing.index_artifacts import IndexArtifactStore

    storage = MagicMock()
    storage.INDEXES_PREFIX = "indexes"
    storage.list_objects.return_value = [
        ("indexes/v20260101-010000-b/index.faiss", 1),
        ("indexes/not-a-stamped-version/index.faiss", 1),
        ("indexes/v20260101-010000-a/index.faiss", 1),
        ("indexes/v20251231-235959/index.faiss", 1),
    ]
    storage.build_index_key.side_effect = lambda version, filename: f"indexes/{version}/{filename}"
    storage.exists.side_effect = lambda key: (
        key
        in {
            "indexes/v20260101-010000-a/index.faiss",
            "indexes/v20260101-010000-a/mapping.json",
        }
    )

    store = IndexArtifactStore(storage=storage, cache_dir=tmp_path)

    assert store.list_versions() == [
        "not-a-stamped-version",
        "v20251231-235959",
        "v20260101-010000-a",
        "v20260101-010000-b",
    ]
    assert store.has_required_artifacts("v20260101-010000-a") is True
    assert store.has_required_artifacts("v20260101-010000-b") is False


def test_artifact_store_reports_cached_text_index_by_index_file(
    tmp_path: Path,
) -> None:
    from activities.indexing.index_artifacts import IndexArtifactStore

    storage = MagicMock()
    cached = tmp_path / "v1"
    cached.mkdir()
    (cached / "text_index.faiss").write_bytes(b"cached")

    store = IndexArtifactStore(storage=storage, cache_dir=tmp_path)

    assert store.has_text_artifacts_cached("v1") is True


def test_artifact_store_cleans_uploaded_objects_on_upload_failure(tmp_path: Path) -> None:
    from activities.indexing.index_artifacts import IndexArtifactStore

    storage = MagicMock()
    storage.INDEXES_PREFIX = "indexes"
    storage.build_index_key.side_effect = lambda version, filename: f"indexes/{version}/{filename}"
    storage.upload_file.side_effect = [None, RuntimeError("upload failed")]
    storage.list_objects.return_value = [
        ("indexes/v1/index.faiss", 1),
        ("indexes/v1/mapping.json", 1),
    ]
    first = tmp_path / "index.faiss"
    second = tmp_path / "mapping.json"
    first.write_bytes(b"index")
    second.write_text("{}", encoding="utf-8")

    store = IndexArtifactStore(storage=storage, cache_dir=tmp_path / "cache")

    try:
        store.upload_files("v1", [first, second])
    except RuntimeError as exc:
        assert str(exc) == "upload failed"
    else:
        raise AssertionError("upload failure should propagate")

    deleted = {call.args[0] for call in storage.delete.call_args_list}
    assert deleted == {"indexes/v1/index.faiss", "indexes/v1/mapping.json"}


def test_vector_index_search_maps_faiss_rows_to_image_ids() -> None:
    from activities.indexing.faiss_vectors import FaissVectorIndex

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    vector_index = FaissVectorIndex.build(
        embeddings=embeddings,
        image_ids=[101, 202],
        index_type="flat",
    )

    results = vector_index.search(np.array([1.0, 0.0], dtype=np.float32), k=2)

    assert results[0][0] == 101
    assert vector_index.get_vector_by_image_id(202) is not None
    assert vector_index.get_vector_by_image_id(303) is None


def test_active_index_catalog_swap_marks_only_requested_build_active(
    db_session,
) -> None:
    from activities.indexing.index_catalog import ActiveIndexCatalog

    from tests.factories import create_index_build

    create_index_build(session=db_session, version="v1", is_active=True)
    create_index_build(session=db_session, version="v2", is_active=False)
    db_session.flush()

    catalog = ActiveIndexCatalog()

    catalog.swap_to_version("v2", db_session)

    builds = {
        build.version: build.is_active for build in catalog.list_builds_newest_first(db_session)
    }
    assert builds == {"v1": False, "v2": True}


def test_active_index_catalog_garbage_collection_retains_newest_and_skips_active(
    db_session,
) -> None:
    from activities.indexing.index_catalog import ActiveIndexCatalog

    from tests.factories import create_index_build

    base = datetime(2026, 1, 1, 12, 0, 0)
    create_index_build(
        session=db_session,
        version="v-new",
        is_active=False,
        created_at=base + timedelta(minutes=2),
    )
    create_index_build(
        session=db_session,
        version="v-active-old",
        is_active=True,
        created_at=base + timedelta(minutes=1),
    )
    create_index_build(
        session=db_session,
        version="v-old",
        is_active=False,
        created_at=base,
    )
    db_session.flush()
    artifacts = MagicMock()

    removed = ActiveIndexCatalog().garbage_collect(
        db_session,
        artifacts=artifacts,
        retain_versions=1,
    )

    remaining = {
        build.version for build in ActiveIndexCatalog().list_builds_newest_first(db_session)
    }
    assert removed == ["v-old"]
    assert remaining == {"v-new", "v-active-old"}
    artifacts.delete_stored_version.assert_called_once_with("v-old")
    artifacts.delete_cached_version.assert_called_once_with("v-old")
