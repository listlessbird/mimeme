from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment
from tests.support.storage import Memory

from mimeme import index
from mimeme.config import Settings
from mimeme.index import activity as index_activity
from mimeme.index.activity import Activities


def _build() -> index.Build:
    return index.Build(
        job_id="rebuild-activity",
        version="v2-g2-activity",
        target_generation=2,
        model="test/embed",
        index_type="flat",
        dimension=2,
        encoder=index.Encoder(repo="encoder", revision="rev", variant="model.onnx"),
        embeddings=[index.Embedding(image_id=1, image_key="embeddings/1.npy")],
    )


class _Success:
    async def build(self, request: index.Build, *, progress=None) -> index.Result:  # noqa: ANN001
        assert progress is not None
        await progress("native", 0.5)
        return index.Result(outcome="empty")


class _Invalid:
    async def build(self, request: index.Build, *, progress=None) -> index.Result:  # noqa: ANN001
        raise ValueError("invalid build request")


@dataclass
class _Env:
    index: object
    db: object = object()
    artifacts: object = field(default_factory=Memory)
    settings: Settings = field(default_factory=Settings)
    search: object = object()


async def test_build_activity_heartbeats_compute_progress() -> None:
    heartbeats: list[object] = []
    activity_env = ActivityEnvironment()
    activity_env.on_heartbeat = lambda *details: heartbeats.extend(details)

    result = await activity_env.run(Activities(_Env(index=_Success())).build, _build())  # type: ignore[arg-type]

    assert result.outcome == "empty"
    assert heartbeats == [{"phase": "native", "progress": 0.5, "version": "v2-g2-activity"}]


async def test_build_activity_marks_invalid_input_non_retryable(monkeypatch) -> None:  # noqa: ANN001
    async def fail(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    async def cleanup(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    monkeypatch.setattr(index_activity.index, "fail", fail)
    monkeypatch.setattr(index_activity.index, "cleanup_incomplete", cleanup)
    with pytest.raises(ApplicationError) as raised:
        await ActivityEnvironment().run(Activities(_Env(index=_Invalid())).build, _build())  # type: ignore[arg-type]

    assert raised.value.non_retryable is True
    assert raised.value.type == "ValueError"


async def test_activate_activity_heartbeats_while_work_is_running(monkeypatch) -> None:  # noqa: ANN001
    async def activate(*args, **kwargs) -> index.Activated:  # noqa: ANN002, ANN003
        await asyncio.sleep(0)
        return index.Activated(version="v2")

    monkeypatch.setattr(index_activity.index, "activate", activate)
    heartbeats: list[object] = []
    activity_env = ActivityEnvironment()
    activity_env.on_heartbeat = lambda *details: heartbeats.extend(details)

    result = await activity_env.run(
        Activities(_Env(index=_Success())).activate,  # type: ignore[arg-type]
        index.ActivateInput(
            job_id="rebuild-activity",
            target_generation=2,
            result=index.Result(outcome="empty"),
        ),
    )

    assert result.version == "v2"
    assert heartbeats == [{"phase": "activate", "job_id": "rebuild-activity"}]
