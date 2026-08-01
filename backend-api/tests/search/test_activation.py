from __future__ import annotations

import pytest

from mimeme import search
from mimeme.search.activation import activate, reconcile


class _Activation:
    def __init__(self, *, serving: str | None = None) -> None:
        self.serving = serving
        self.calls: list[tuple[str, str]] = []

    async def load(self, generation: search.Load) -> search.Loaded:
        self.calls.append(("load", generation.version))
        return search.Loaded(
            version=generation.version,
            embed_model="test/embed",
            dimension=2,
            image_count=3,
            faiss_version="1.13.2",
            onnxruntime_version="1.27.0",
            encoder_revision="rev-1",
        )

    async def switch(self, version: str) -> search.Status:
        self.calls.append(("switch", version))
        self.serving = version
        return await self.status()

    async def rollback(self, failed_version: str) -> search.Status:
        self.calls.append(("rollback", failed_version))
        self.serving = "v1"
        return await self.status()

    async def status(self) -> search.Status:
        return search.Status(ready=self.serving is not None, serving_version=self.serving)


def _generation(version: str = "v2") -> search.Load:
    digest = "0" * 64
    return search.Load(
        version=version,
        files=[
            search.File(name="index.faiss", key="i", sha256=digest),
            search.File(name="mapping.json", key="m", sha256=digest),
            search.File(name="metadata.json", key="d", sha256=digest),
        ],
        encoder=search.Encoder(repo="repo", revision="rev", variant="model.onnx", threads=1),
    )


async def test_failed_database_commit_rolls_compute_back() -> None:
    remote = _Activation(serving="v1")

    async def fail(_: search.Loaded) -> None:
        raise RuntimeError("database commit failed")

    with pytest.raises(RuntimeError, match="database commit failed"):
        await activate(_generation(), activation=remote, commit=fail)

    assert remote.calls == [("load", "v2"), ("switch", "v2"), ("rollback", "v2")]
    assert remote.serving == "v1"


async def test_startup_reconciliation_loads_database_desired_generation() -> None:
    remote = _Activation(serving=None)

    status = await reconcile(_generation(), activation=remote)

    assert status.serving_version == "v2"
    assert remote.calls == [("load", "v2"), ("switch", "v2")]


async def test_reconciliation_is_a_noop_when_compute_already_matches() -> None:
    remote = _Activation(serving="v2")

    status = await reconcile(_generation(), activation=remote)

    assert status.serving_version == "v2"
    assert remote.calls == []
