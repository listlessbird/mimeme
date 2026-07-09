from __future__ import annotations

from httpx import AsyncClient


async def test_async_client_can_call_app_without_lifespan_side_effects(
    async_client: AsyncClient,
) -> None:
    response = await async_client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["openapi"].startswith("3.")
