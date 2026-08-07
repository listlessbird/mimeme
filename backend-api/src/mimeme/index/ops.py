from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError

from mimeme import inference, search, storage
from mimeme.config import Settings
from mimeme.db import Db
from mimeme.db.schema import Job, JobStatus, JobType
from mimeme.index import pack, rule
from mimeme.index.client import Client
from mimeme.index.model import (
    Activated,
    ActivateInput,
    Backfilled,
    Build,
    Encoder,
    Manifest,
    Prepared,
    PrepareInput,
    Result,
    Sealed,
    SealInput,
    Trigger,
)
from mimeme.index.store import Store
from mimeme.job.model import ClaimOwnership, StateMissing
from mimeme.job.store import Store as JobStore

_MANIFEST_MAX = 1024 * 1024
_BACKFILL_BATCH = 1000
_TERMINAL = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


async def prepare(db: Db, settings: Settings, input: PrepareInput) -> Prepared:
    now = datetime.now(UTC)
    async with db.write_session() as session:
        jobs = JobStore(session)
        view = await jobs.lock_freshness()
        claim = view.active_claim
        owns_claim = claim is not None and claim.job_id == input.job_id
        if claim is not None:
            owner = await session.get(Job, claim.job_id)
            expired = (
                claim.claimed_at + timedelta(minutes=settings.index.rebuild_claim_timeout_minutes)
                <= now
            )
            if owner is None or owner.status in _TERMINAL or expired:
                if owner is not None and owner.status not in _TERMINAL:
                    await jobs.fail_rebuild(
                        claim.job_id,
                        f"Rebuild claim expired; reclaimed by {input.trigger.value}",
                    )
                await jobs.release(job_id=claim.job_id)
                view = await jobs.lock_freshness()
                owns_claim = False
        if view.active_claim is not None and not owns_claim:
            return Prepared(decision="busy", job_id=input.job_id)
        if not view.is_stale and not input.force:
            if input.trigger is Trigger.MANUAL and input.job_id is not None:
                active = await jobs.active_build_stats()
                await jobs.complete_rebuild(
                    job_id=input.job_id,
                    version=active.version if active else "",
                    num_vectors=active.num_vectors if active else 0,
                    dimension=active.dimension if active else 0,
                    removed_versions=[],
                    text_num_vectors=None,
                    skipped=True,
                    skip_reason="already_current",
                    message="Index already current",
                )
            return Prepared(decision="clean", job_id=input.job_id)
        if not input.force and not rule.settled(
            now=now,
            last_dirty_at=view.last_dirty_at,
            last_reconciled_at=view.last_reconciled_at,
            settle=timedelta(minutes=settings.index.rebuild_settle_minutes),
            max_stale=timedelta(hours=settings.index.rebuild_max_stale_hours),
        ):
            if input.trigger is Trigger.MANUAL and input.job_id is not None:
                await jobs.complete_rebuild(
                    job_id=input.job_id,
                    version="",
                    num_vectors=0,
                    dimension=0,
                    removed_versions=[],
                    text_num_vectors=None,
                    skipped=True,
                    skip_reason="dirty_stream_moving",
                    message="Rebuild deferred until the dirty stream settles",
                )
            return Prepared(decision="deferred", job_id=input.job_id)
        job_id = input.job_id
        if input.trigger is Trigger.SCHEDULED:
            assert job_id is not None
            if await session.get(Job, job_id) is None:
                await jobs.add_job(
                    job_id=job_id,
                    job_type=JobType.REBUILD_INDEX,
                    workflow_id=input.workflow_id,
                )
        assert job_id is not None
        if owns_claim:
            target_generation = view.rebuild_target_generation
        else:
            claim_result = await jobs.claim(job_id=job_id, force=input.force, now=now)
            target_generation = claim_result.view.rebuild_target_generation
        assert target_generation is not None
        await jobs.mark_running(job_id)
        snapshot = await Store(session).snapshot(
            model=input.model,
            target_generation=target_generation,
        )
    planned_reads = pack.reads(snapshot.embeddings)
    version = _version(job_id, target_generation)
    return Prepared(
        decision="build",
        job_id=job_id,
        build=Build(
            job_id=job_id,
            version=version,
            target_generation=target_generation,
            model=input.model,
            index_type=input.index_type,
            dimension=snapshot.dimension,
            native_threads=settings.index.build_threads,
            encoder=Encoder(
                repo=settings.search.encoder_repo,
                revision=settings.search.encoder_revision,
                variant=settings.search.encoder_variant,
                threads=settings.search.encoder_threads,
            ),
            embeddings=snapshot.embeddings,
            planned_reads=planned_reads,
        ),
    )


async def backfill_text_presence(
    db: Db,
    artifacts: storage.Store,
    *,
    model: str,
    batch: int = _BACKFILL_BATCH,
) -> Backfilled:
    seen = 0
    marked = 0
    pending: list[str] = []
    async for info in artifacts.list(prefix=inference.embedding_prefix(model)):
        if not inference.is_text_embedding_key(info.object.key):
            continue
        seen += 1
        pending.append(inference.image_embedding_key_of(info.object.key))
        if len(pending) >= batch:
            marked += await _mark_present(db, model=model, image_keys=pending)
            pending = []
    marked += await _mark_present(db, model=model, image_keys=pending)
    async with db.write_session() as session:
        absent = await Store(session).mark_text_absent(model=model)
    return Backfilled(model=model, text_objects=seen, marked_present=marked, marked_absent=absent)


async def _mark_present(db: Db, *, model: str, image_keys: list[str]) -> int:
    if not image_keys:
        return 0
    async with db.write_session() as session:
        return await Store(session).mark_text_present(model=model, image_keys=image_keys)


async def build(client: Client, request: Build, *, progress=None) -> Result:  # noqa: ANN001
    return await client.build(request, progress=progress)


async def seal(db: Db, client: Client, settings: Settings, input: SealInput) -> Sealed:
    return await pack.seal(
        db,
        client,
        job_id=input.job_id,
        model=input.model,
        shard_rows=settings.index.shard_rows,
        max_shards=settings.index.seal_max_shards,
        min_rows=settings.index.seal_min_rows,
    )


async def activate(
    db: Db,
    artifacts: storage.Store,
    remote: search.Activation,
    input: ActivateInput,
    *,
    retain: int = 5,
) -> Activated:
    if input.cancelled or input.error is not None:
        await _record_failure(db, input)
        await cleanup_incomplete(artifacts, version=None, protect=set())
        return Activated(version="")
    assert input.result is not None
    if input.result.outcome == "empty":
        async with db.write_session() as session:
            await Store(session).reconcile_empty(
                job_id=input.job_id, target_generation=input.target_generation
            )
        return Activated(version="")
    assert input.result.manifest is not None
    manifest = await validate(artifacts, input.result.manifest)
    load = _load(manifest)

    async def commit(_loaded: search.Loaded) -> None:
        async with db.write_session() as session:
            await Store(session).activate(job_id=input.job_id, manifest=manifest)

    if await _activation_matches(db, manifest):
        await search.reconcile(load, activation=remote)
    else:
        await search.activate(load, activation=remote, commit=commit)
    status = await _confirm_activation(db, remote, manifest)
    protect = {version for version in (status.serving_version, status.retained_version) if version}
    removed = await collect(db, artifacts, protect=protect, retain=retain)
    return Activated(version=manifest.version, removed_versions=removed)


async def reconcile(
    db: Db,
    artifacts: storage.Store,
    remote: search.Activation,
) -> search.Status | None:
    async with db.read_session() as session:
        version = await Store(session).active_version()
    if version is None:
        status = await remote.status()
        if (
            status.serving_version is None
            and status.candidate_version is None
            and status.retained_version is None
        ):
            return status
        return await remote.clear()
    raw = await artifacts.read_bytes(
        storage.Object(f"indexes/{version}/complete.json"), max_bytes=_MANIFEST_MAX
    )
    manifest = await validate(artifacts, Manifest.model_validate_json(raw))
    return await search.reconcile(_load(manifest), activation=remote)


async def validate(artifacts: storage.Store, expected: Manifest) -> Manifest:
    raw = await artifacts.read_bytes(storage.Object(expected.complete_key), max_bytes=_MANIFEST_MAX)
    try:
        actual = Manifest.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid completeness manifest: {exc}") from exc
    if actual != expected:
        raise ValueError("published completeness manifest does not match build result")
    for file in actual.files:
        info = await artifacts.stat(storage.Object(file.key))
        if info is None or info.length != file.length:
            raise ValueError(f"artifact is missing or has wrong length: {file.key}")
        if info.checksum is not None and info.checksum.value != file.sha256:
            raise ValueError(f"artifact checksum mismatch: {file.key}")
    return actual


async def collect(
    db: Db,
    artifacts: storage.Store,
    *,
    protect: set[str],
    retain: int,
) -> list[str]:
    async with db.read_session() as session:
        versions = await Store(session).removable(protect=protect, retain=retain)
    for version in versions:
        await _delete_prefix(artifacts, f"indexes/{version}/")
    async with db.write_session() as session:
        await Store(session).forget(versions)
    return versions


async def cleanup_incomplete(
    artifacts: storage.Store, *, version: str | None, protect: set[str]
) -> None:
    if version is None or version in protect:
        return
    complete = await artifacts.stat(storage.Object(f"indexes/{version}/complete.json"))
    if complete is None:
        await _delete_prefix(artifacts, f"indexes/{version}/")


async def fail(db: Db, *, job_id: str, error: str, cancelled: bool) -> None:
    await _record_failure(
        db,
        ActivateInput(
            job_id=job_id,
            target_generation=0,
            error=None if cancelled else error,
            cancelled=cancelled,
        ),
    )


async def _record_failure(db: Db, input: ActivateInput) -> None:
    async with db.write_session() as session:
        try:
            await Store(session).fail(
                job_id=input.job_id,
                error=input.error or "cancelled",
                cancelled=input.cancelled,
            )
        except (ClaimOwnership, StateMissing):
            pass


async def _delete_prefix(artifacts: storage.Store, prefix: str) -> None:
    objects = [info.object async for info in artifacts.list(prefix=prefix)]
    await asyncio.gather(*(artifacts.delete(obj) for obj in objects))


def _version(job_id: str, generation: int) -> str:
    digest = hashlib.sha256(job_id.encode()).hexdigest()[:12]
    return f"v2-g{generation}-{digest}"


def _load(manifest: Manifest) -> search.Load:
    return search.Load(
        version=manifest.version,
        files=[
            search.File(name=file.name, key=file.key, sha256=file.sha256) for file in manifest.files
        ],
        encoder=search.Encoder(
            repo=manifest.encoder.repo,
            revision=manifest.encoder.revision,
            variant=manifest.encoder.variant,
            threads=manifest.encoder.threads,
        ),
    )


async def _activation_matches(db: Db, manifest: Manifest) -> bool:
    async with db.read_session() as session:
        indexes = Store(session)
        jobs = JobStore(session)
        return (
            await indexes.active_version() == manifest.version
            and (await jobs.freshness()).active_generation == manifest.target_generation
        )


async def _confirm_activation(
    db: Db,
    remote: search.Activation,
    manifest: Manifest,
) -> search.Status:
    if not await _activation_matches(db, manifest):
        raise RuntimeError("database active generation does not match the activated manifest")
    status = await remote.status()
    if status.serving_version != manifest.version:
        raise RuntimeError(
            f"search serves {status.serving_version!r}, expected {manifest.version!r}"
        )
    return status
