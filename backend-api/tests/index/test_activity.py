from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace

import pytest
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment
from tests.support.storage import Memory

from mimeme import index, storage
from mimeme.config import Settings
from mimeme.index import activity as index_activity
from mimeme.index import ops as index_ops
from mimeme.index import rule
from mimeme.index.activity import Activities


def _plan() -> index.BuildPlan:
    return index.BuildPlan(
        job_id="rebuild-activity",
        version="v2-g2-activity",
        target_generation=2,
        model="test/embed",
        index_type="flat",
        dimension=2,
        encoder=index.Encoder(repo="encoder", revision="rev", variant="model.onnx"),
        embeddings_key=index_ops.plan_key("v2-g2-activity"),
        num_embeddings=1,
    )


async def _seed(artifacts: Memory) -> index.BuildPlan:
    plan = _plan()
    await artifacts.put_bytes(
        storage.Object(plan.embeddings_key),
        index.EmbeddingManifest(
            version=plan.version,
            dimension=plan.dimension,
            embeddings=[index.Embedding(image_id=1, image_key="embeddings/1.npy")],
        )
        .model_dump_json()
        .encode(),
        content_type="application/json",
    )
    return plan


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

    env = _Env(index=_Success())
    plan = await _seed(env.artifacts)

    result = await activity_env.run(Activities(env).build, plan)  # type: ignore[arg-type]

    assert result.outcome == "empty"
    assert heartbeats == [{"phase": "native", "progress": 0.5, "version": "v2-g2-activity"}]


async def test_build_activity_marks_invalid_input_non_retryable(monkeypatch) -> None:  # noqa: ANN001
    async def fail(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    async def cleanup(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    monkeypatch.setattr(index_activity.index, "fail", fail)
    monkeypatch.setattr(index_activity.index, "cleanup_incomplete", cleanup)
    env = _Env(index=_Invalid())
    plan = await _seed(env.artifacts)
    with pytest.raises(ApplicationError) as raised:
        await ActivityEnvironment().run(Activities(env).build, plan)  # type: ignore[arg-type]

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


async def test_seal_activity_skips_when_another_seal_holds_the_lock(monkeypatch) -> None:  # noqa: ANN001
    async def busy(*args, **kwargs) -> index.Sealed:  # noqa: ANN002, ANN003
        raise index_activity.pack.Busy("another seal holds the pack lock")

    monkeypatch.setattr(index_activity.index, "seal", busy)

    result = await ActivityEnvironment().run(
        Activities(_Env(index=_Success())).seal,  # type: ignore[arg-type]
        index.SealInput(job_id="rebuild-activity", model="test/embed"),
    )

    assert (result.shards, result.rows) == (0, 0)


async def test_seal_activity_retries_before_giving_up_the_claim(monkeypatch) -> None:  # noqa: ANN001
    released: list[str] = []

    async def unreachable(*args, **kwargs) -> index.Sealed:  # noqa: ANN002, ANN003
        raise RuntimeError("compute unreachable")

    async def fail(db, *, job_id: str, error: str, cancelled: bool) -> None:  # noqa: ANN001
        released.append(job_id)

    async def cleanup(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    monkeypatch.setattr(index_activity.index, "seal", unreachable)
    monkeypatch.setattr(index_activity.index, "fail", fail)
    monkeypatch.setattr(index_activity.index, "cleanup_incomplete", cleanup)
    request = index.SealInput(job_id="rebuild-activity", model="test/embed")

    first = ActivityEnvironment()
    first.info = replace(first.info, attempt=1)
    with pytest.raises(RuntimeError, match="compute unreachable"):
        await first.run(Activities(_Env(index=_Success())).seal, request)  # type: ignore[arg-type]
    assert released == []

    last = ActivityEnvironment()
    last.info = replace(last.info, attempt=rule.SEAL_MAX_ATTEMPTS)
    with pytest.raises(RuntimeError, match="compute unreachable"):
        await last.run(Activities(_Env(index=_Success())).seal, request)  # type: ignore[arg-type]
    assert released == ["rebuild-activity"]
