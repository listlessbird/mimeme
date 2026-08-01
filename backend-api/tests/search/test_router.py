from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mimeme import search
from mimeme.search import router
from mimeme.search.run import Projection
from mimeme.shared.config import Settings
from mimeme.shared.services.media_url import MediaUrlResolver


class _Client:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.queries: list[search.Query] = []

    async def query(
        self, query: search.Query, *, count: int, cursor: str | None = None
    ) -> search.Batch:
        if self.error:
            raise self.error
        self.queries.append(query)
        return search.Batch(
            candidates=[search.Candidate(image_id=2, score=0.9)],
            exhausted=True,
            version="v1",
        )

    async def status(self) -> search.Status:
        return search.Status(ready=True, serving_version="v1")

    async def close(self) -> None:
        pass


class _Rows:
    async def fetch(self, image_ids: list[int]) -> dict[int, Projection]:
        return {
            2: Projection(
                id=2,
                sha256="abc",
                media_key="images/2.jpg",
                caption="cat",
                ocr_text=None,
                width=800,
                height=600,
            )
        }


def _app(client: _Client) -> FastAPI:
    app = FastAPI()
    app.state.env = SimpleNamespace(settings=Settings(app_env="development"), search=client)
    app.state.limiter = SimpleNamespace()
    app.include_router(router.router)
    app.dependency_overrides[router.get_client] = lambda: client
    app.dependency_overrides[router.get_rows] = _Rows
    app.dependency_overrides[router.get_media_urls] = lambda: MediaUrlResolver("https://media.test")
    return app


def test_search_route_awaits_the_shared_client_and_preserves_response() -> None:
    remote = _Client()
    with TestClient(_app(remote)) as client:
        response = client.get("/search?q=cat&mode=hybrid&limit=5")

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == 2
    assert response.json()["results"][0]["url"] == "https://media.test/images/2.jpg"
    assert remote.queries[0].mode == "hybrid"


def test_search_route_maps_compute_unavailability() -> None:
    with TestClient(_app(_Client(error=search.Unavailable("compute restarting")))) as client:
        response = client.get("/search?q=cat")

    assert response.status_code == 503
    assert response.json()["detail"] == "compute restarting"


def test_similar_route_maps_missing_query_image() -> None:
    with TestClient(_app(_Client(error=search.NotFound("image not indexed")))) as client:
        response = client.get("/search/similar/99")

    assert response.status_code == 404
