from __future__ import annotations

import httpx
import pytest

from mimeme import search
from mimeme.search.remote import Remote


async def test_remote_uses_typed_gateway_contract() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/search/query":
            return httpx.Response(
                200,
                json={
                    "candidates": [{"image_id": 7, "score": 0.75}],
                    "cursor": None,
                    "exhausted": True,
                    "version": "v1",
                },
            )
        return httpx.Response(
            200,
            json={
                "ready": True,
                "serving_version": "v1",
                "candidate_version": None,
                "retained_version": None,
                "embed_model": "test/embed",
                "encoder_revision": "rev-1",
                "detail": None,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = Remote(http, base_url="http://compute.test")
        batch = await client.query(search.Query(text="cat"), count=32)
        status = await client.status()

    assert batch.candidates[0].image_id == 7
    assert status.serving_version == "v1"
    assert [request.url.path for request in requests] == [
        "/v1/search/query",
        "/v1/search/status",
    ]


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("search_unavailable", search.Unavailable),
        ("search_loading", search.Loading),
        ("search_incompatible", search.Incompatible),
        ("search_invalid", search.Invalid),
        ("search_not_found", search.NotFound),
        ("search_stale", search.Stale),
        ("search_failed", search.Failed),
    ],
)
async def test_remote_maps_typed_failures(code: str, error: type[Exception]) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": {"code": code, "message": "nope"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = Remote(http, base_url="http://compute.test")
        with pytest.raises(error, match="nope"):
            await client.status()


async def test_remote_maps_transport_timeout_to_unavailable() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = Remote(http, base_url="http://compute.test")
        with pytest.raises(search.Unavailable, match="slow"):
            await client.status()
