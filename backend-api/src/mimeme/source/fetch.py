from __future__ import annotations

import asyncio
import hashlib
import time
from types import TracebackType
from typing import Any, cast

from mimeme.source.http import Http
from mimeme.source.model import FetchRequest, Retryable
from mimeme.storage.model import Object
from mimeme.storage.store import Store

_CHECKPOINT_PREFIX = "source-fetch"
_MAX_CHECKPOINT_BYTES = 5 * 1024 * 1024


class Fetcher:
    """One discovery attempt's transports, cookies, pacing, and URL cache."""

    def __init__(
        self,
        json_http: Http,
        *,
        delay_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        retries: int = 3,
        impersonate: str = "chrome",
        artifacts: Store | None = None,
        checkpoint_id: str | None = None,
    ) -> None:
        self._json_http = json_http
        self._delay = delay_seconds
        self._timeout = timeout_seconds
        self._retries = retries
        self._impersonate = impersonate
        self._artifacts = artifacts
        self._checkpoint_id = checkpoint_id
        self._manager: Any = None
        self._session = None
        self._last_request_at = 0.0
        self._html_by_url: dict[str, bytes] = {}
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> Fetcher:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._manager is not None:
            await self._manager.__aexit__(exc_type, exc, traceback)

    async def json(self, request: FetchRequest) -> dict | None:
        response = await self._json_http.fetch(request)
        return response.raw if response.success else None

    async def html(self, url: str) -> bytes:
        cached = self._html_by_url.get(url)
        if cached is not None:
            return cached
        async with self._lock:
            cached = self._html_by_url.get(url)
            if cached is not None:
                return cached
            checkpoint = self._checkpoint_object(url)
            if checkpoint is not None and self._artifacts is not None:
                info = await self._artifacts.stat(checkpoint)
                if info is not None:
                    body = await self._artifacts.read_bytes(
                        checkpoint, max_bytes=_MAX_CHECKPOINT_BYTES
                    )
                    self._html_by_url[url] = body
                    return body
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._delay:
                await asyncio.sleep(self._delay - elapsed)
            session = await self._html_session()
            try:
                response = await session.get(
                    url,
                    timeout=self._timeout,
                    retries=self._retries,
                    stealthy_headers=True,
                )
            except Exception as exc:
                raise Retryable(f"source fetch failed for {url}: {exc}") from exc
            self._last_request_at = time.monotonic()
            if response.status == 404:
                raise PageNotFound(url)
            if response.status != 200:
                raise Retryable(f"source fetch HTTP {response.status} for {url}")
            body = bytes(response.body)
            if len(body) > _MAX_CHECKPOINT_BYTES:
                raise Retryable(f"source page exceeds checkpoint limit for {url}")
            if checkpoint is not None and self._artifacts is not None:
                await self._artifacts.put_bytes(checkpoint, body, content_type="text/html")
            self._html_by_url[url] = body
            return body

    async def cleanup(self) -> None:
        if self._artifacts is None or self._checkpoint_id is None:
            return
        await cleanup_checkpoint(self._artifacts, self._checkpoint_id)

    def _checkpoint_object(self, url: str) -> Object | None:
        if self._checkpoint_id is None:
            return None
        digest = hashlib.sha256(url.encode()).hexdigest()
        return Object(f"{_CHECKPOINT_PREFIX}/{self._checkpoint_id}/{digest}.html")

    async def _html_session(self):
        if self._session is None:
            from scrapling.fetchers import FetcherSession

            self._manager = FetcherSession(
                impersonate=cast(Any, self._impersonate),
                stealthy_headers=True,
                timeout=self._timeout,
                retries=self._retries,
            )
            self._session = await self._manager.__aenter__()
        return self._session


class PageNotFound(Exception):
    pass


async def cleanup_checkpoint(artifacts: Store, checkpoint_id: str) -> None:
    prefix = f"{_CHECKPOINT_PREFIX}/{checkpoint_id}/"
    async for info in artifacts.list(prefix=prefix):
        await artifacts.delete(info.object)
