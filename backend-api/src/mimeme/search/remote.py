from __future__ import annotations

from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from mimeme.search.error import (
    Error,
    Failed,
    Incompatible,
    Invalid,
    Loading,
    NotFound,
    Stale,
    Unavailable,
)
from mimeme.search.model import (
    Batch,
    CandidateRequest,
    Load,
    Loaded,
    Query,
    Rollback,
    Status,
    Switch,
)

_T = TypeVar("_T", bound=BaseModel)
_ERRORS: dict[str, type[Error]] = {
    error.code: error
    for error in (Unavailable, Loading, Incompatible, Invalid, NotFound, Stale, Failed)
}


class Remote:
    def __init__(self, http: httpx.AsyncClient, *, base_url: str) -> None:
        self._http = http
        self._base = base_url.rstrip("/")

    async def query(self, query: Query, *, count: int, cursor: str | None = None) -> Batch:
        request = CandidateRequest(query=query, count=count, cursor=cursor)
        return await self._request("POST", "/v1/search/query", Batch, json=request.model_dump())

    async def status(self) -> Status:
        return await self._request("GET", "/v1/search/status", Status)

    async def load(self, generation: Load) -> Loaded:
        return await self._request("POST", "/v1/search/load", Loaded, json=generation.model_dump())

    async def switch(self, version: str) -> Status:
        return await self._request(
            "POST", "/v1/search/switch", Status, json=Switch(version=version).model_dump()
        )

    async def rollback(self, failed_version: str) -> Status:
        return await self._request(
            "POST",
            "/v1/search/rollback",
            Status,
            json=Rollback(failed_version=failed_version).model_dump(),
        )

    async def close(self) -> None:
        # The environment owns the shared HTTP client.
        return None

    async def _request(
        self,
        method: str,
        path: str,
        model: type[_T],
        *,
        json: dict | None = None,
    ) -> _T:
        try:
            response = await self._http.request(method, f"{self._base}{path}", json=json)
        except httpx.HTTPError as exc:
            raise Unavailable(f"search compute unavailable: {exc}") from exc
        if response.is_error:
            _raise_remote(response)
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise Failed(f"invalid search compute response: {exc}") from exc


def _raise_remote(response: httpx.Response) -> None:
    code = "search_failed"
    message = f"search compute returned HTTP {response.status_code}"
    try:
        detail = response.json().get("detail")
        if isinstance(detail, dict):
            code = str(detail.get("code", code))
            message = str(detail.get("message", message))
        elif detail:
            message = str(detail)
    except ValueError:
        pass
    raise _ERRORS.get(code, Failed)(message)
