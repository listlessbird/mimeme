from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from mimeme.search.document import PROJECTION_VERSION, SearchDocument

SCHEMA_VERSION = 1
TOKENIZER = "porter unicode61"
FIELDS = (
    "titles",
    "tags",
    "ocr",
    "captions",
    "classifiers",
    "provenance",
    "descriptions",
)
WEIGHTS = (4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0)


@dataclass(frozen=True)
class Built:
    sha256: str
    length: int
    count: int
    sqlite_version: str


class Index:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection


def build(path: Path, documents: list[SearchDocument]) -> Built:
    if path.exists():
        raise ValueError(f"BM25 output already exists: {path}")
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE metadata ("
            "schema_version INTEGER NOT NULL, projection_version INTEGER NOT NULL, "
            "tokenizer TEXT NOT NULL, weights TEXT NOT NULL, document_count INTEGER NOT NULL, "
            "sqlite_version TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE documents USING fts5("
            "image_id UNINDEXED, titles, tags, ocr, captions, classifiers, provenance, "
            f"descriptions, tokenize='{TOKENIZER}')"
        )
        with connection:
            connection.executemany(
                "INSERT INTO documents "
                "(image_id, titles, tags, ocr, captions, classifiers, provenance, descriptions) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (_row(value) for value in documents),
            )
            connection.execute(
                "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?)",
                (
                    SCHEMA_VERSION,
                    PROJECTION_VERSION,
                    TOKENIZER,
                    ",".join(str(weight) for weight in WEIGHTS),
                    len(documents),
                    sqlite3.sqlite_version,
                ),
            )
            connection.execute("INSERT INTO documents(documents) VALUES ('optimize')")
    finally:
        connection.close()
    payload = path.read_bytes()
    return Built(
        sha256=hashlib.sha256(payload).hexdigest(),
        length=len(payload),
        count=len(documents),
        sqlite_version=sqlite3.sqlite_version,
    )


def open(path: Path, *, count: int, sqlite_version: str) -> Index:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        integrity = connection.execute("PRAGMA quick_check").fetchone()
        if integrity != ("ok",):
            raise ValueError("BM25 SQLite integrity check failed")
        metadata = connection.execute(
            "SELECT schema_version, projection_version, tokenizer, weights, "
            "document_count, sqlite_version FROM metadata"
        ).fetchone()
        expected = (
            SCHEMA_VERSION,
            PROJECTION_VERSION,
            TOKENIZER,
            ",".join(str(weight) for weight in WEIGHTS),
            count,
            sqlite_version,
        )
        if metadata != expected:
            raise ValueError("BM25 metadata does not match its generation descriptor")
        return Index(connection)
    except Exception:
        connection.close()
        raise


def query(index: Index, text: str, *, depth: int) -> list[int]:
    terms = _terms(text)
    if not terms or depth < 1:
        return []
    match = " ".join(f'"{term}"' for term in terms)
    weights = ", ".join(str(weight) for weight in WEIGHTS)
    rows = index.connection.execute(
        f"SELECT image_id FROM documents WHERE documents MATCH ? "  # noqa: S608
        f"ORDER BY bm25(documents, 0, {weights}), CAST(image_id AS INTEGER) LIMIT ?",
        (match, depth),
    )
    return [int(image_id) for (image_id,) in rows]


def close(index: Index) -> None:
    index.connection.close()


def _row(value: SearchDocument) -> tuple[int, str, str, str, str, str, str, str]:
    return (
        value.image_id,
        " ".join(value.titles),
        " ".join(value.tags),
        " ".join(value.ocr_texts),
        " ".join(value.captions),
        " ".join((*value.categories, *value.types)),
        " ".join((*value.origins, *value.years)),
        " ".join(value.descriptions),
    )


def _terms(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for character in text.casefold():
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(dict.fromkeys(tokens))
