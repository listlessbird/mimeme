from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from time import perf_counter
from typing import Protocol

import httpx

from mimeme import inference, storage
from mimeme.db import Db
from mimeme.db.schema import DuplicateReason, IngestStage
from mimeme.ingest import rule
from mimeme.ingest.facts import Images
from mimeme.ingest.model import (
    Input,
    InvalidImage,
    Result,
    Retryable,
    Source,
    Staged,
)
from mimeme.ingest.store import ExistingImage, Store
from mimeme.job import ops as job_ops
from mimeme.job.store import Store as JobStore


class Deps(Protocol):
    db: Db
    media: storage.Store
    artifacts: storage.Store
    http: httpx.AsyncClient
    inference: inference.Client
    image_facts: Images


async def run(env: Deps, input: Input) -> Result:
    started = perf_counter()
    timings: dict[str, float] = {}
    staging: str | None = None
    try:
        await job_ops.record_stage(env.db, input.item_id, IngestStage.DOWNLOADING)
        acquire_started = perf_counter()
        data, staging = await _acquire(env, input)
        timings["download_ms"] = round((perf_counter() - acquire_started) * 1000, 2)
        await job_ops.record_stage(env.db, input.item_id, IngestStage.PROCESSING)
        facts = await env.image_facts.inspect(staging, role="artifacts")
        async with env.db.read_session() as session:
            context = await Store(session).inference_context(input.item_id)
        result = await _resolve(env, input, data, staging, facts, context, timings=timings)
    except InvalidImage as exc:
        result = await _fail(env, input, exc)
    if staging is not None:
        await _cleanup_staging(env, staging)
    return result.model_copy(
        update={**timings, "total_ms": round((perf_counter() - started) * 1000, 2)}
    )


async def finish(env: Deps, job_id: str) -> tuple[int, int, int]:
    async with env.db.write_session() as session:
        store = Store(session)
        processed, failed, duplicates = await store.sweep_and_count(job_id, "item did not complete")
        await JobStore(session).complete_ingest(
            job_id=job_id, processed=processed, failed=failed, duplicates=duplicates
        )
    return processed, failed, duplicates


async def _acquire(env: Deps, input: Input) -> tuple[bytes | None, str]:
    source = input.source
    if isinstance(source, Staged):
        return None, source.artifact_key
    data = await _download(env.http, source.url)
    key = rule.staging_key(input.item_id)
    try:
        await env.artifacts.put(
            storage.Object(key),
            _once(data),
            length=len(data),
            content_type="application/octet-stream",
            checksum=storage.Checksum.of(data),
        )
    except storage.Error as exc:
        raise _map_storage(exc) from exc
    return data, key


async def _download(http: httpx.AsyncClient, url: str) -> bytes:
    headers = {
        "user-agent": rule.DOWNLOAD_USER_AGENT,
        "accept": rule.IMAGE_ACCEPT_HEADER,
        "accept-language": "en-US,en;q=0.9",
    }
    try:
        async with http.stream("GET", url, headers=headers, follow_redirects=True) as resp:
            if resp.status_code >= 400:
                if rule.is_terminal_http_status(resp.status_code):
                    raise InvalidImage(f"download failed: HTTP {resp.status_code} for {url}")
                raise Retryable(f"download retryable: HTTP {resp.status_code} for {url}")
            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise InvalidImage(f"non_image_content_type:{content_type} for {url}")
            buffer = bytearray()
            async for chunk in resp.aiter_bytes():
                buffer += chunk
                if len(buffer) > rule.MAX_IMAGE_BYTES:
                    raise InvalidImage(f"image exceeds {rule.MAX_IMAGE_BYTES} bytes for {url}")
            return bytes(buffer)
    except httpx.TimeoutException as exc:
        raise Retryable(f"download timeout for {url}: {exc}") from exc
    except httpx.TransportError as exc:
        raise Retryable(f"download transport error for {url}: {exc}") from exc


async def _resolve(
    env: Deps,
    input: Input,
    data: bytes | None,
    staging: str,
    facts,
    context,
    *,
    timings: dict[str, float],
) -> Result:
    async with env.db.read_session() as session:
        store = Store(session)
        phash_dedup_allowed = await store.phash_dedup_allowed(input.item_id)
        sha_id = await store.find_by_sha(facts.sha256)
        phash_match = None if sha_id is not None else await store.find_phash_match(facts.phash)
    if sha_id is not None:
        return await _duplicate(
            env, input, sha_id, DuplicateReason.SHA256, context, timings=timings
        )
    if phash_match is not None and phash_dedup_allowed:
        return await _duplicate(
            env, input, phash_match.image_id, DuplicateReason.PHASH, context, timings=timings
        )

    payload = data if data is not None else await _read_staged(env, staging)
    media_key = rule.canonical_media_key(
        sha256=facts.sha256, dataset=input.dataset, image_format=facts.format
    )
    try:
        info = await env.media.put(
            storage.Object(media_key),
            _once(payload),
            length=len(payload),
            content_type=rule.content_type_for(facts.format),
            checksum=storage.Checksum(value=facts.sha256),
        )
    except storage.Error as exc:
        raise _map_storage(exc) from exc

    async with env.db.write_session() as session:
        store = Store(session)
        await store.acquire_dedup_lock()
        sha_id = await store.find_by_sha(facts.sha256)
        if sha_id is not None:
            resolved = (sha_id, DuplicateReason.SHA256)
        else:
            phash_match = await store.find_phash_match(facts.phash)
            if phash_match is not None and phash_dedup_allowed:
                resolved = (phash_match.image_id, DuplicateReason.PHASH)
            else:
                image_id = await store.insert_canonical(
                    facts=facts,
                    dataset=input.dataset,
                    filename=_filename(input.source),
                    s3_key=media_key,
                    etag=info.etag,
                    file_size=len(payload),
                )
                resolved = None

    if resolved is not None:
        return await _duplicate(
            env, input, resolved[0], resolved[1], context, timings=timings
        )
    return await _complete_new(
        env,
        input,
        image_id,
        media_key,
        facts.sha256,
        context,
        timings=timings,
        similar_image_id=(phash_match.image_id if phash_match is not None else None),
        phash_distance=(phash_match.distance if phash_match is not None else None),
    )


async def _complete_new(
    env: Deps,
    input: Input,
    image_id: int,
    media_key: str,
    sha256: str,
    context,
    timings: dict[str, float],
    *,
    similar_image_id: int | None,
    phash_distance: int | None,
) -> Result:
    await _annotate_and_embed(
        env,
        input,
        image_id=image_id,
        media_key=media_key,
        sha256=sha256,
        do_annotation=True,
        do_embedding=True,
        caption=None,
        ocr_text=None,
        context=context,
        timings=timings,
    )
    await job_ops.mark_item_done(
        env.db,
        input.item_id,
        image_id,
        similar_image_id=similar_image_id,
        phash_distance=phash_distance,
    )
    await job_ops.record_stage(env.db, input.item_id, IngestStage.COMPLETE)
    return Result(item_id=input.item_id, outcome="processed", image_id=image_id)


async def _duplicate(
    env: Deps,
    input: Input,
    image_id: int,
    reason: DuplicateReason,
    context,
    *,
    timings: dict[str, float],
) -> Result:
    async with env.db.write_session() as session:
        view: ExistingImage = await Store(session).duplicate_view(image_id)
    context_hash = rule.text_sha256(context.model_dump_json()) if context is not None else None
    caption_context_changed = context is not None and (
        view.caption_context_sha256 != context_hash
        or view.caption_prompt_version != inference.CAPTION_PROMPT_VERSION
    )
    desired_text = rule.compose_search_text(context, view.existing_caption, view.existing_ocr_text)
    facts_changed = context is not None and (
        view.embed_text_sha256 != rule.text_sha256(desired_text)
        or view.embed_recipe_version != rule.EMBED_RECIPE_VERSION
    )
    await _annotate_and_embed(
        env,
        input,
        image_id=image_id,
        media_key=view.s3_key,
        sha256=view.sha256,
        do_annotation=view.needs_annotation or caption_context_changed,
        do_embedding=view.needs_embedding or facts_changed or caption_context_changed,
        caption=view.existing_caption,
        ocr_text=view.existing_ocr_text,
        context=context,
        timings=timings,
    )
    await job_ops.record_stage(env.db, input.item_id, IngestStage.DEDUPED)
    await job_ops.mark_item_done(
        env.db,
        input.item_id,
        image_id,
        duplicate_reason=reason,
        duplicate_of_image_id=image_id,
    )
    return Result(
        item_id=input.item_id, outcome="duplicate", image_id=image_id, duplicate_reason=reason
    )


async def _annotate_and_embed(
    env: Deps,
    input: Input,
    *,
    image_id: int,
    media_key: str,
    sha256: str,
    do_annotation: bool,
    do_embedding: bool,
    caption: str | None,
    ocr_text: str | None,
    context,
    timings: dict[str, float],
) -> None:
    if do_annotation:
        await job_ops.record_stage(env.db, input.item_id, IngestStage.ANNOTATING)
        annotation_started = perf_counter()
        annotation = await _annotate(env, image_id, media_key, context)
        timings["annotation_ms"] = round((perf_counter() - annotation_started) * 1000, 2)
        await job_ops.save_annotations(
            env.db,
            image_id=image_id,
            caption=annotation.caption,
            caption_model=annotation.caption_model,
            ocr_text=annotation.ocr_text,
            ocr_model=annotation.ocr_model,
            caption_context_sha256=(
                rule.text_sha256(context.model_dump_json()) if context is not None else None
            ),
            caption_prompt_version=(
                inference.CAPTION_PROMPT_VERSION if context is not None else None
            ),
        )
        caption, ocr_text = annotation.caption, annotation.ocr_text

    if do_embedding:
        await job_ops.record_stage(env.db, input.item_id, IngestStage.EMBEDDING)
        text = rule.compose_search_text(context, caption, ocr_text)
        text_hash = rule.text_sha256(text)
        embedding_started = perf_counter()
        embedding = await _embed(env, image_id, media_key, text, sha256, input.dataset)
        timings["embedding_ms"] = round((perf_counter() - embedding_started) * 1000, 2)
        if embedding is not None:
            await job_ops.save_embedding(
                env.db,
                image_id=embedding.image_id,
                model=embedding.model,
                dimension=embedding.dimension,
                image_embedding_key=embedding.image_embedding_key,
                text_embedding_key=embedding.text_embedding_key,
                text_sha256=text_hash,
                recipe_version=rule.EMBED_RECIPE_VERSION,
            )


async def _annotate(env: Deps, image_id: int, media_key: str, context) -> inference.Annotation:
    try:
        return await env.inference.annotate(
            inference.Input(image_id=image_id, media_key=media_key, context=context)
        )
    except (inference.Unavailable, inference.Timeout) as exc:
        raise Retryable(f"annotate unavailable: {exc}") from exc
    except inference.Invalid as exc:
        raise InvalidImage(f"annotate failed: {exc}") from exc


async def _embed(
    env: Deps, image_id: int, media_key: str, text: str, sha256: str, dataset: str | None
) -> inference.Embedding | None:
    batch = inference.Batch(
        items=[
            inference.Item(
                image_id=image_id, media_key=media_key, text=text, sha256=sha256, dataset=dataset
            )
        ],
        dataset=dataset,
    )
    try:
        result = await env.inference.embed(batch)
    except (inference.Unavailable, inference.Timeout) as exc:
        raise Retryable(f"embed unavailable: {exc}") from exc
    except inference.Invalid as exc:
        raise InvalidImage(f"embed failed: {exc}") from exc
    results = result.results
    return results[0] if results else None


async def _fail(env: Deps, input: Input, exc: InvalidImage) -> Result:
    error = rule.truncate_error(str(exc))
    await job_ops.mark_item_failed(env.db, input.item_id, error)
    return Result(item_id=input.item_id, outcome="failed", error=error)


async def _read_staged(env: Deps, key: str) -> bytes:
    try:
        return await env.artifacts.read_bytes(storage.Object(key), max_bytes=rule.MAX_IMAGE_BYTES)
    except storage.Error as exc:
        raise _map_storage(exc) from exc


async def _cleanup_staging(env: Deps, key: str) -> None:
    with suppress(storage.Error):
        await env.artifacts.delete(storage.Object(key))


def _map_storage(exc: storage.Error) -> Exception:
    if isinstance(exc, (storage.Timeout, storage.Unavailable, storage.Integrity)):
        return Retryable(str(exc))
    if isinstance(exc, storage.Missing):
        return InvalidImage(f"missing staged object: {exc}")
    return InvalidImage(str(exc))


def _filename(source: Source) -> str | None:
    if isinstance(source, Staged):
        return Path(source.artifact_key).name or None
    parsed = httpx.URL(source.url)
    name = Path(parsed.path).name
    return name or None


async def _once(data: bytes) -> AsyncIterator[bytes]:
    yield data
