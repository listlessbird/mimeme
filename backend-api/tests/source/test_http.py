from __future__ import annotations

import httpx
import pytest

from mimeme.source.http import Http
from mimeme.source.model import FetchRequest, Retryable


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_success_returns_raw() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"memes": []})

    async with _client(handler) as client:
        response = await Http(client).fetch(FetchRequest(url="https://x/ok"))
    assert response.success and response.raw == {"memes": []}


async def test_terminal_4xx_is_unsuccessful_no_raise() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with _client(handler) as client:
        response = await Http(client).fetch(FetchRequest(url="https://x/missing"))
    assert not response.success and response.status_code == 404


async def test_retryable_5xx_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with _client(handler) as client:
        with pytest.raises(Retryable):
            await Http(client).fetch(FetchRequest(url="https://x/down"))


async def test_retryable_429_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    async with _client(handler) as client:
        with pytest.raises(Retryable):
            await Http(client).fetch(FetchRequest(url="https://x/ratelimited"))


async def test_timeout_raises_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with _client(handler) as client:
        with pytest.raises(Retryable):
            await Http(client).fetch(FetchRequest(url="https://x/slow"))


async def test_invalid_json_is_unsuccessful() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    async with _client(handler) as client:
        response = await Http(client).fetch(FetchRequest(url="https://x/bad"))
    assert not response.success and response.error is not None


async def test_non_dict_json_is_unsuccessful() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    async with _client(handler) as client:
        response = await Http(client).fetch(FetchRequest(url="https://x/list"))
    assert not response.success and "unexpected_json_shape" in (response.error or "")
