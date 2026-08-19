from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mimeme import env as env_module
from mimeme.config import ComputeConfig, Settings


def test_compute_inference_gateway_reads_independently(monkeypatch) -> None:
    monkeypatch.setenv("COMPUTE_GATEWAY_URL", "http://pi:8010")
    monkeypatch.setenv("COMPUTE_INFERENCE_GATEWAY_URL", "http://gpu:8010")

    config = ComputeConfig(_env_file=None)

    assert config.gateway_url == "http://pi:8010"
    assert config.inference_gateway_url == "http://gpu:8010"


def test_tumblr_api_key_reads_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("TUMBLR_API_KEY", "test-tumblr-key")

    settings = Settings(_env_file=None)

    assert settings.tumblr_api_key is not None
    assert settings.tumblr_api_key.get_secret_value() == "test-tumblr-key"


async def test_partial_startup_closes_resources_in_reverse_order(monkeypatch) -> None:
    closed: list[str] = []

    class FakeDb:
        async def close(self) -> None:
            closed.append("db")

    class FakeStore:
        async def close(self) -> None:
            closed.append("media")

    opens = 0

    async def open_store(_config):
        nonlocal opens
        opens += 1
        if opens == 2:
            raise RuntimeError("artifact storage unavailable")
        return FakeStore()

    connect = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr(env_module, "Db", lambda _config: FakeDb())
    monkeypatch.setattr(env_module.Client, "connect", connect)
    monkeypatch.setattr(env_module.storage.S3, "open", open_store)

    with pytest.raises(RuntimeError, match="artifact storage unavailable"):
        await env_module.Env.create(Settings())

    assert closed == ["media", "db"]
    assert connect.await_args.kwargs["namespace"] == Settings().temporal.namespace
