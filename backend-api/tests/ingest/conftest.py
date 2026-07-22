from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import httpx
import pytest
from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncConnection

from mimeme import inference
from mimeme.ingest.facts import Facts
from tests.job.conftest import PoolDb, SavepointDb, _async_url
from tests.support.storage import Memory


@pytest.fixture()
def db(async_db_connection: AsyncConnection) -> SavepointDb:
    return SavepointDb(async_db_connection)


@pytest.fixture()
async def pool_db(db_engine: Engine):
    if db_engine.dialect.name != "postgresql":
        pytest.skip("advisory-lock tests require PostgreSQL")
    pool = PoolDb(_async_url(db_engine))
    yield pool
    await pool.close()


class FakeInference:
    """Deterministic `inference.Client` that annotates and embeds every image."""

    def __init__(
        self,
        *,
        caption: str = "a caption",
        ocr_text: str = "ocr text",
        dimension: int = 8,
        embed_model: str = "google/siglip2",
        fail_annotation: Exception | None = None,
        fail_embedding: Exception | None = None,
        embed_ok: bool = True,
    ) -> None:
        self.caption = caption
        self.ocr_text = ocr_text
        self.dimension = dimension
        self.embed_model = embed_model
        self.fail_annotation = fail_annotation
        self.fail_embedding = fail_embedding
        self.embed_ok = embed_ok
        self.annotate_calls: list[inference.Input] = []
        self.embed_calls: list[inference.Batch] = []

    async def annotate(self, input: inference.Input, *, progress=None) -> inference.Annotation:
        self.annotate_calls.append(input)
        if self.fail_annotation is not None:
            raise self.fail_annotation
        return inference.Annotation(
            image_id=input.image_id,
            caption=self.caption,
            caption_model="fake-caption",
            ocr_text=self.ocr_text,
            ocr_model="fake-ocr",
        )

    async def embed(self, batch: inference.Batch, *, progress=None) -> inference.BatchResult:
        self.embed_calls.append(batch)
        if self.fail_embedding is not None:
            raise self.fail_embedding
        items: list[inference.Ok | inference.Failed] = []
        for item in batch.items:
            if not self.embed_ok:
                items.append(inference.Failed(image_id=item.image_id, error="embed failed"))
                continue
            key = inference.image_embedding_key(
                sha256=item.sha256, model=self.embed_model, dataset=item.dataset
            )
            items.append(
                inference.Ok(
                    embedding=inference.Embedding(
                        image_id=item.image_id,
                        image_embedding_key=key,
                        text_embedding_key=inference.text_embedding_key(key),
                        model=self.embed_model,
                        dimension=self.dimension,
                    )
                )
            )
        return inference.BatchResult(items=items)

    async def ready(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class FakeImages:
    """Fake compute image-facts client. Returns configured facts per key."""

    def __init__(self, default: Facts | None = None) -> None:
        self._by_key: dict[str, Facts] = {}
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def set(self, key: str, facts: Facts) -> None:
        self._by_key[key] = facts

    async def inspect(self, key: str, *, role: str = "artifacts") -> Facts:
        self.calls.append((key, role))
        facts = self._by_key.get(key, self._default)
        if facts is None:
            raise AssertionError(f"no fake facts configured for {key}")
        return facts


@dataclass
class FakeEnv:
    db: object
    media: Memory = field(default_factory=Memory)
    artifacts: Memory = field(default_factory=Memory)
    http: httpx.AsyncClient | None = None
    inference: FakeInference = field(default_factory=FakeInference)
    image_facts: FakeImages = field(default_factory=FakeImages)
    temporal: object | None = None


def facts_for(data: bytes, *, phash: str, image_format: str = "png", mode: str = "RGB") -> Facts:
    return Facts(
        sha256=hashlib.sha256(data).hexdigest(),
        phash=phash,
        width=16,
        height=16,
        format=image_format,
        mode=mode,
    )


def image_http(data: bytes, *, content_type: str = "image/png", status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        if status >= 400:
            return httpx.Response(status)
        return httpx.Response(status, content=data, headers={"content-type": content_type})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture()
def png_bytes() -> bytes:
    # Not a real PNG; the compute image child is faked, so raw bytes are enough.
    return b"\x89PNG\r\n\x1a\n" + b"fake-image-payload" * 4
