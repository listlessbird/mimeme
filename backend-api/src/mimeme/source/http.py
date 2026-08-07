from __future__ import annotations

import httpx

from mimeme.source import rule
from mimeme.source.model import FetchRequest, RawResponse, Retryable


class Http:
    """Async fetch transport over one long-lived, process-scoped HTTPX client.

    Terminal 4xx and malformed bodies return an unsuccessful ``RawResponse``.
    Transient failures (5xx, 408/425/429, timeouts, transport errors) raise
    ``Retryable`` so the coarse discover activity retries the whole fetch."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def fetch(self, request: FetchRequest) -> RawResponse:
        headers = {**rule._DEFAULT_HEADERS, **request.headers}
        try:
            response = await self._client.request(
                str(request.method),
                request.url,
                headers=headers,
                follow_redirects=True,
                timeout=request.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise Retryable(f"fetch timeout for {request.url}: {exc}") from exc
        except httpx.TransportError as exc:
            raise Retryable(f"fetch transport error for {request.url}: {exc}") from exc

        status = response.status_code
        if status >= 400:
            if rule.is_terminal_http_status(status):
                return RawResponse(
                    success=False,
                    status_code=status,
                    error=f"HTTP {status} for {request.url}",
                )
            raise Retryable(f"HTTP {status} for {request.url}")

        try:
            raw = response.json()
        except ValueError as exc:
            return RawResponse(success=False, status_code=status, error=f"invalid_json:{exc}")

        if not isinstance(raw, dict):
            return RawResponse(
                success=False,
                status_code=status,
                error=f"unexpected_json_shape:{type(raw).__name__}",
            )

        return RawResponse(success=True, status_code=status, raw=raw)
