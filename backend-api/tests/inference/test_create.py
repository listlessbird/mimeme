from __future__ import annotations

import httpx

from mimeme import inference
from mimeme.config import ComputeConfig, Settings
from mimeme.inference.local import Local


async def test_local_inference_uses_the_dedicated_gateway() -> None:
    settings = Settings(
        compute=ComputeConfig(
            gpu_backend="local",
            gateway_url="http://pi:8010",
            inference_gateway_url="http://gpu:8010",
        )
    )
    http = httpx.AsyncClient()

    try:
        client = inference.create(settings, http)
        assert isinstance(client, Local)
        assert client._base == "http://gpu:8010"  # noqa: SLF001
    finally:
        await http.aclose()


async def test_local_inference_falls_back_to_the_main_gateway() -> None:
    settings = Settings(compute=ComputeConfig(gpu_backend="local", gateway_url="http://pi:8010"))
    http = httpx.AsyncClient()

    try:
        client = inference.create(settings, http)
        assert isinstance(client, Local)
        assert client._base == "http://pi:8010"  # noqa: SLF001
    finally:
        await http.aclose()
