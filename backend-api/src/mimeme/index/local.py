from __future__ import annotations

import asyncio

import httpx

from mimeme.compute.model import JobState
from mimeme.index.client import Progress
from mimeme.index.gateway import Failed
from mimeme.index.model import Build, BuildSpec, Result, Seal, SealResult, SealSpec


class Local:
    def __init__(
        self, http: httpx.AsyncClient, *, base_url: str, poll_interval_s: float = 5.0
    ) -> None:
        self._http = http
        self._base = base_url.rstrip("/")
        self._poll = min(max(poll_interval_s, 0.01), 5.0)

    async def seal(self, request: Seal) -> SealResult:
        state = await self._run(
            f"seal-{request.job_id}", SealSpec(seal=request).model_dump(), progress=None
        )
        return SealResult.model_validate(state.result)

    async def build(self, request: Build, *, progress: Progress | None = None) -> Result:
        state = await self._run(
            request.job_id, BuildSpec(build=request).model_dump(), progress=progress
        )
        return Result.model_validate(state.result)

    async def _run(self, job_id: str, payload: dict, *, progress: Progress | None) -> JobState:
        url = f"{self._base}/v1/jobs/{job_id}"
        try:
            response = await self._http.put(url, json=payload)
            response.raise_for_status()
            state = JobState.model_validate(response.json())
            while state.status in ("queued", "running"):
                if progress is not None:
                    await progress(state.phase or state.status, state.progress)
                await asyncio.sleep(self._poll)
                response = await self._http.get(url)
                response.raise_for_status()
                state = JobState.model_validate(response.json())
        except asyncio.CancelledError:
            try:
                await self._http.delete(url)
            finally:
                raise
        except (httpx.HTTPError, ValueError) as exc:
            raise Failed(f"index compute unavailable: {exc}") from exc
        if state.status != "succeeded":
            raise Failed(state.error or f"index compute job ended as {state.status}")
        return state

    async def close(self) -> None:
        return None
