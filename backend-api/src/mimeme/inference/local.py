from __future__ import annotations

import asyncio
import hashlib

import httpx

from mimeme.compute.model import (
    AnnotateResult,
    AnnotateSpec,
    EmbedResult,
    EmbedSpec,
    EmbedSpecItem,
    JobState,
)
from mimeme.inference.client import Progress
from mimeme.inference.model import (
    Annotation,
    Batch,
    BatchResult,
    Embedding,
    Failed,
    Input,
    Invalid,
    Ok,
    Timeout,
    Unavailable,
    image_embedding_key,
    text_embedding_key,
)


class Local:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        base_url: str,
        embed_model: str,
        poll_interval_s: float = 5.0,
    ) -> None:
        self._http = http
        self._base = base_url.rstrip("/")
        self._embed_model = embed_model
        self._poll = poll_interval_s

    async def ready(self) -> bool:
        try:
            resp = await self._http.get(f"{self._base}/v1/roles/inference/ready")
        except httpx.HTTPError:
            return False
        if resp.status_code != 200:
            return False
        return bool(resp.json().get("ok"))

    async def annotate(self, input: Input, *, progress: Progress | None = None) -> Annotation:
        spec = AnnotateSpec(media_key=input.media_key, length=input.length, context=input.context)
        job_id = _annotate_job_id(input)
        state = await self._drive(job_id, spec.model_dump(), progress)
        result = AnnotateResult.model_validate(state.result)
        return Annotation(
            image_id=input.image_id,
            caption=result.caption,
            caption_model=result.caption_model,
            ocr_text=result.ocr_text,
            ocr_model=result.ocr_model,
        )

    async def embed(self, batch: Batch, *, progress: Progress | None = None) -> BatchResult:
        spec_items: list[EmbedSpecItem] = []
        for item in batch.items:
            dataset = item.dataset or batch.dataset
            image_key = image_embedding_key(
                sha256=item.sha256, model=self._embed_model, dataset=dataset
            )
            spec_items.append(
                EmbedSpecItem(
                    image_id=item.image_id,
                    media_key=item.media_key,
                    text=item.text,
                    sha256=item.sha256,
                    image_key=image_key,
                    text_key=text_embedding_key(image_key),
                )
            )
        spec = EmbedSpec(model=self._embed_model, items=spec_items)
        job_id = _embed_job_id(spec_items)
        state = await self._drive(job_id, spec.model_dump(), progress)
        result = EmbedResult.model_validate(state.result)
        items: list[Ok | Failed] = []
        for entry in result.items:
            if entry.ok and entry.image_key and entry.text_key and entry.model and entry.dimension:
                items.append(
                    Ok(
                        embedding=Embedding(
                            image_id=entry.image_id,
                            image_embedding_key=entry.image_key,
                            text_embedding_key=entry.text_key,
                            model=entry.model,
                            dimension=entry.dimension,
                        )
                    )
                )
            else:
                items.append(
                    Failed(image_id=entry.image_id, error=entry.error or "embedding failed")
                )
        return BatchResult(items=items)

    async def _drive(self, job_id: str, spec: dict, progress: Progress | None) -> JobState:
        url = f"{self._base}/v1/jobs/{job_id}"
        try:
            state = await self._put(url, spec)
            while state.status in ("queued", "running"):
                if progress is not None:
                    await progress(state.phase or state.status, state.progress)
                wait_started = asyncio.get_running_loop().time()
                state = await self._get(url, wait_s=self._poll)
                # Older gateways ignore wait_s and answer immediately. Yield briefly
                # when that happens so rolling deployments cannot create a hot loop.
                elapsed = asyncio.get_running_loop().time() - wait_started
                fallback_delay = min(self._poll, 0.05)
                if state.status in ("queued", "running") and elapsed < fallback_delay:
                    await asyncio.sleep(fallback_delay - elapsed)
        except asyncio.CancelledError:
            await self._delete(url)
            raise
        if state.status == "succeeded":
            return state
        if state.status == "cancelled":
            raise Invalid(f"job {job_id} cancelled")
        if state.error and state.error.startswith("child_dead:"):
            raise Unavailable(state.error)
        raise Invalid(state.error or f"job {job_id} failed")

    async def _put(self, url: str, spec: dict) -> JobState:
        return _state(await self._request("PUT", url, json=spec))

    async def _get(self, url: str, *, wait_s: float) -> JobState:
        return _state(await self._request("GET", url, params={"wait_s": wait_s}))

    async def _delete(self, url: str) -> None:
        try:
            await self._request("DELETE", url)
        except (Unavailable, Timeout):
            pass

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json: dict | None = None,
        params: dict[str, float] | None = None,
    ) -> httpx.Response:
        try:
            resp = await self._http.request(method, url, json=json, params=params)
        except httpx.TimeoutException as exc:
            raise Timeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise Unavailable(str(exc)) from exc
        if resp.status_code >= 500:
            raise Unavailable(f"compute {resp.status_code}: {resp.text}")
        if resp.status_code >= 400:
            raise Invalid(f"compute {resp.status_code}: {resp.text}")
        return resp

    async def close(self) -> None:
        return None


def _state(resp: httpx.Response) -> JobState:
    try:
        return JobState.model_validate(resp.json())
    except Exception as exc:
        raise Unavailable(f"malformed compute response: {exc}") from exc


def _annotate_job_id(input: Input) -> str:
    context = input.context.model_dump_json() if input.context is not None else ""
    digest = hashlib.sha256(f"{input.media_key}|{input.length}|{context}".encode()).hexdigest()[:32]
    return f"annotate-{digest}"


def _embed_job_id(items: list[EmbedSpecItem]) -> str:
    payload = "|".join(f"{item.image_id}:{item.media_key}:{item.text}" for item in items)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:32]
    return f"embed-{digest}"
