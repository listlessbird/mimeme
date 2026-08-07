from __future__ import annotations

from mimeme import search
from mimeme.api.routers.health import _check_inference, _check_search


class _Client:
    def __init__(self, status: search.Status | Exception) -> None:
        self._status = status

    async def status(self) -> search.Status:
        if isinstance(self._status, Exception):
            raise self._status
        return self._status


async def test_api_readiness_requires_a_serving_search_generation() -> None:
    assert await _check_search(search_client := _Client(search.Status(ready=False))) is False
    search_client._status = search.Status(ready=True, serving_version="v1")
    assert await _check_search(search_client) is True


async def test_api_readiness_is_degraded_when_compute_is_unavailable() -> None:
    assert await _check_search(_Client(search.Unavailable("down"))) is False


class _Inference:
    async def ready(self) -> bool:
        return False


async def test_api_readiness_requires_inference_compute() -> None:
    assert await _check_inference(_Inference()) is False
