from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import modal

from mimeme import release
from mimeme.inference import bge
from mimeme.inference.client import Progress
from mimeme.inference.model import (
    ANNOTATION_CONTRACT_VERSION,
    Annotation,
    Batch,
    BatchResult,
    Embedding,
    Failed,
    Input,
    Invalid,
    Ok,
    Unavailable,
    image_embedding_key,
)


class Modal:
    def __init__(self, *, app_name: str, embed_model: str, poll_interval_s: float = 5.0) -> None:
        self._app_name = app_name
        self._embed_model = embed_model
        self._poll = poll_interval_s
        self._vision: Any = None
        self._embedding: Any = None
        self._bge: Any = None
        self._release_info: Any = None

    def _vision_cls(self) -> Any:
        if self._vision is None:
            self._vision = modal.Cls.from_name(self._app_name, "VisionService")
        return self._vision

    def _embedding_cls(self) -> Any:
        if self._embedding is None:
            self._embedding = modal.Cls.from_name(self._app_name, "EmbeddingService")
        return self._embedding

    def _bge_cls(self) -> Any:
        if self._bge is None:
            self._bge = modal.Cls.from_name(self._app_name, "BgeService")
        return self._bge

    async def ready(self) -> bool:
        try:
            self._vision_cls()
            self._embedding_cls()
            self._bge_cls()
            if self._release_info is None:
                self._release_info = modal.Function.from_name(self._app_name, "release_info")
            info = await self._release_info.remote.aio()
        except Exception:
            return False
        return info == {
            "release_id": release.ID,
            "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
        }

    async def annotate(self, input: Input, *, progress: Progress | None = None) -> Annotation:
        instance = self._vision_cls()()
        kwargs: dict[str, Any] = {"media_key": input.media_key, "length": input.length}
        if input.context is not None:
            kwargs["context"] = input.context.model_dump(mode="json")
        result = await self._call(instance.annotate_image, progress, **kwargs)
        return Annotation(
            image_id=input.image_id,
            caption=result["caption"],
            caption_model=result["caption_model"],
            ocr_text=result["ocr_text"],
            ocr_model=result["ocr_model"],
        )

    async def embed(self, batch: Batch, *, progress: Progress | None = None) -> BatchResult:
        items = []
        for item in batch.items:
            dataset = item.dataset or batch.dataset
            image_key = image_embedding_key(
                sha256=item.sha256, model=self._embed_model, dataset=dataset
            )
            items.append(
                {
                    "image_id": item.image_id,
                    "media_key": item.media_key,
                    "sha256": item.sha256,
                    "image_key": image_key,
                }
            )
        instance = self._embedding_cls()()
        result = await self._call(
            instance.embed_batch, progress, items=items, model=self._embed_model
        )
        out: list[Ok | Failed] = []
        for entry in result["items"]:
            if entry.get("ok"):
                out.append(
                    Ok(
                        embedding=Embedding(
                            image_id=entry["image_id"],
                            image_embedding_key=entry["image_key"],
                            model=entry["model"],
                            dimension=entry["dimension"],
                        )
                    )
                )
            else:
                out.append(
                    Failed(image_id=entry["image_id"], error=entry.get("error") or "embed failed")
                )
        return BatchResult(items=out)

    async def embed_bge(
        self, batch: bge.EncodeBatch, *, progress: Progress | None = None
    ) -> bge.EncodedBatch:
        instance = self._bge_cls()()
        raw = await self._call(
            instance.encode_batch,
            progress,
            batch=batch.model_dump(mode="json"),
        )
        result = bge.EncodedBatch.model_validate(raw)
        bge.validate_result(batch, result)
        return result

    async def _call(self, method: Any, progress: Progress | None, **kwargs: Any) -> dict:
        try:
            fc = await method.spawn.aio(**kwargs)
        except modal.exception.Error as exc:
            raise Unavailable(str(exc)) from exc

        async def _poller() -> None:
            while True:
                await asyncio.sleep(self._poll)
                if progress is not None:
                    await progress("remote", 0.5)

        poller = asyncio.create_task(_poller())
        try:
            result = await fc.get.aio()
        except asyncio.CancelledError:
            with suppress(BaseException):
                await fc.cancel.aio(terminate_containers=True)
            raise
        except modal.exception.Error as exc:
            raise Unavailable(str(exc)) from exc
        finally:
            poller.cancel()
            with suppress(BaseException):
                await poller
        if not isinstance(result, dict):
            raise Invalid(f"unexpected modal result type: {type(result).__name__}")
        return result

    async def close(self) -> None:
        return None
