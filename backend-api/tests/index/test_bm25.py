from __future__ import annotations

import sqlite3

import pytest

from mimeme.index import bm25
from mimeme.search.document import SearchDocument


def _documents() -> list[SearchDocument]:
    return [
        SearchDocument(
            image_id=1,
            titles=("Distracted boyfriend", "Man looking at another woman"),
            tags=("reaction",),
            captions=("A wandering eye",),
        ),
        SearchDocument(
            image_id=2,
            ocr_texts=("deploy broke friday",),
            descriptions=("A rare quokka release failure",),
        ),
        SearchDocument(image_id=3, categories=("Animals",), origins=("Unicode café",)),
    ]


def test_build_open_and_query_fielded_documents(tmp_path) -> None:
    path = tmp_path / "bm25.sqlite3"
    built = bm25.build(path, _documents())

    index = bm25.open(path, count=3, sqlite_version=sqlite3.sqlite_version)
    try:
        assert bm25.query(index, "distracted boyfriend", depth=10) == [1]
        assert bm25.query(index, "quokka", depth=10) == [2]
        assert bm25.query(index, "deploy friday", depth=10) == [2]
        assert bm25.query(index, "café", depth=10) == [3]
        assert bm25.query(index, "missing", depth=10) == []
    finally:
        bm25.close(index)
    assert built.count == 3
    assert built.length == path.stat().st_size
    assert len(built.sha256) == 64
    assert not list(tmp_path.glob("*-wal"))
    assert not list(tmp_path.glob("*-shm"))


@pytest.mark.parametrize(
    "query",
    [
        '"distracted" OR quokka',
        "quokka's release",
        "deploy + friday -broken",
        "NEAR(deploy friday)",
        "deploy deploy friday",
        "***",
    ],
)
def test_raw_queries_cannot_become_fts_syntax(tmp_path, query: str) -> None:
    path = tmp_path / "bm25.sqlite3"
    bm25.build(path, _documents())
    index = bm25.open(path, count=3, sqlite_version=sqlite3.sqlite_version)
    try:
        assert isinstance(bm25.query(index, query, depth=10), list)
    finally:
        bm25.close(index)


def test_equivalent_builds_have_equivalent_results_and_empty_corpus_works(tmp_path) -> None:
    results = []
    for number in (1, 2):
        path = tmp_path / f"bm25-{number}.sqlite3"
        built = bm25.build(path, _documents())
        index = bm25.open(path, count=built.count, sqlite_version=built.sqlite_version)
        results.append(bm25.query(index, "reaction", depth=10))
        bm25.close(index)
    assert results == [[1], [1]]

    empty_path = tmp_path / "empty.sqlite3"
    empty = bm25.build(empty_path, [])
    index = bm25.open(empty_path, count=0, sqlite_version=empty.sqlite_version)
    assert bm25.query(index, "anything", depth=10) == []
    bm25.close(index)


def test_open_rejects_descriptor_mismatch_and_corruption(tmp_path) -> None:
    path = tmp_path / "bm25.sqlite3"
    built = bm25.build(path, _documents())
    with pytest.raises(ValueError, match="metadata"):
        bm25.open(path, count=4, sqlite_version=built.sqlite_version)

    path.write_bytes(b"not sqlite")
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        bm25.open(path, count=3, sqlite_version=built.sqlite_version)
