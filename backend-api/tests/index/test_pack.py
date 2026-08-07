from __future__ import annotations

import io

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from tests.factories import (
    create_image,
    create_job,
    create_processing,
    create_search_index_state,
)
from tests.job.conftest import SavepointDb
from tests.support.storage import Memory

from mimeme import index, storage
from mimeme.compute.index import build as build_child
from mimeme.compute.index import pack as pack_child
from mimeme.compute.model import ChildOk
from mimeme.config import Settings
from mimeme.db.schema import JobType, Processing, ProcessingStatus
from mimeme.index import ops, pack, rule
from mimeme.index.gateway import Gateway
from mimeme.index.model import BuildCall, PackCall

_MODEL = "test/embed"
_PREFIX = "embeddings/test_embed"
_DIM = 4


class _Calls:
    def __init__(self) -> None:
        self.calls = 0

    async def call(self, role: str, request: bytes) -> bytes:
        assert role == "index"
        self.calls += 1
        call = PackCall.model_validate_json(request)
        return ChildOk(result=pack_child(call).model_dump()).model_dump_json().encode()


class _Packer:
    def __init__(self, artifacts: Memory, workspace_dir, calls: _Calls | None = None) -> None:  # noqa: ANN001
        self._artifacts = artifacts
        self._workspace_dir = workspace_dir
        self.calls = calls if calls is not None else _Calls()

    async def seal(self, request):  # noqa: ANN001, ANN202
        return await pack.perform(
            self._artifacts,
            self.calls,
            workspace_dir=self._workspace_dir,
            target=request,
        )


async def _seal(db, artifacts, workspace_dir, *, shard_rows, calls=None):  # noqa: ANN001, ANN202
    return await pack.seal(
        db,
        _Packer(artifacts, workspace_dir, calls),
        job_id="seal-test",
        model=_MODEL,
        shard_rows=shard_rows,
    )


class _BuildCalls:
    async def call(self, role: str, request: bytes) -> bytes:
        assert role == "index"
        call = BuildCall.model_validate_json(request)
        return ChildOk(result=build_child(call.build).model_dump()).model_dump_json().encode()


class _FailingUpload(Memory):
    async def put(self, obj, body, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if obj.key.endswith("/text/000000.npy"):
            raise storage.Unavailable("upload interrupted")
        return await super().put(obj, body, **kwargs)


class _FailingSecondShard(Memory):
    async def put(self, obj, body, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if obj.key.endswith("/image/000001.npy"):
            raise storage.Unavailable("upload interrupted")
        return await super().put(obj, body, **kwargs)


def _vector(seed: int) -> np.ndarray:
    return np.array([seed + 1, seed + 2, seed + 3, seed + 4], dtype=np.float32)


def _npy(vector: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, vector)
    return output.getvalue()


async def _seed_corpus(
    db: SavepointDb, run_sync_seed, artifacts: Memory, *, count: int, text: bool = True
) -> list[str]:
    keys = [f"{_PREFIX}/api-ingested/{position:04d}.npy" for position in range(count)]

    def seed(session: Session) -> None:
        for key in keys:
            image = create_image(session=session)
            processing = create_processing(session=session, image=image)
            processing.embed_status = ProcessingStatus.DONE
            processing.embed_model = _MODEL
            processing.embed_dim = _DIM
            processing.embed_s3_key = key
            processing.embed_text_present = text
            session.flush()

    await run_sync_seed(seed)
    for position, key in enumerate(keys):
        await artifacts.put_bytes(
            storage.Object(key), _npy(_vector(position)), content_type="application/octet-stream"
        )
        if text:
            await artifacts.put_bytes(
                storage.Object(key.replace(".npy", "_text.npy")),
                _npy(_vector(position + 100)),
                content_type="application/octet-stream",
            )
    return keys


async def _positions(db: SavepointDb) -> dict[str, tuple[int | None, int | None]]:
    async with db.read_session() as session:
        rows = (
            await session.execute(
                select(Processing.embed_s3_key, Processing.embed_shard, Processing.embed_row)
            )
        ).all()
    return {str(row.embed_s3_key): (row.embed_shard, row.embed_row) for row in rows}


async def test_a_second_seal_is_refused_while_one_holds_the_pack_lock(pool_db) -> None:  # noqa: ANN001
    async with pack._exclusive(pool_db, _MODEL):  # noqa: SLF001
        with pytest.raises(pack.Busy):
            async with pack._exclusive(pool_db, _MODEL):  # noqa: SLF001
                pass


async def test_seals_for_different_models_do_not_block_each_other(pool_db) -> None:  # noqa: ANN001
    async with pack._exclusive(pool_db, _MODEL):  # noqa: SLF001
        async with pack._exclusive(pool_db, "other/embed"):  # noqa: SLF001
            pass


def test_locate_puts_the_two_families_under_one_coordinate() -> None:
    assert pack.locate(_MODEL, 7, text=False) == f"{_PREFIX}/shards/image/000007.npy"
    assert pack.locate(_MODEL, 7, text=True) == f"{_PREFIX}/shards/text/000007.npy"


async def test_plan_seals_only_full_shards_and_leaves_a_bounded_tail(
    index_db: SavepointDb, run_sync_seed
) -> None:
    await _seed_corpus(index_db, run_sync_seed, Memory(), count=7)

    target = await pack.plan(index_db, model=_MODEL, shard_rows=3)

    assert [shard.number for shard in target.shards] == [0, 1]
    assert [len(shard.members) for shard in target.shards] == [3, 3]
    assert target.unsealed == 7
    assert target.tail == 1
    assert target.tail < target.shard_rows


async def test_sealed_rows_reproduce_the_individual_objects_byte_for_byte(
    index_db: SavepointDb, run_sync_seed, tmp_path
) -> None:
    artifacts = Memory()
    keys = await _seed_corpus(index_db, run_sync_seed, artifacts, count=4)
    target = await pack.plan(index_db, model=_MODEL, shard_rows=4)

    sealed = await _seal(index_db, artifacts, tmp_path, shard_rows=4)

    assert (sealed.shards, sealed.rows) == (1, 4)
    images = np.load(
        io.BytesIO(
            await artifacts.read_bytes(storage.Object(target.shards[0].image_key), max_bytes=10**6)
        )
    )
    texts = np.load(
        io.BytesIO(
            await artifacts.read_bytes(storage.Object(target.shards[0].text_key), max_bytes=10**6)
        )
    )
    assert images.shape == texts.shape == (4, _DIM)
    assert images.dtype == texts.dtype == np.float32
    for row, key in enumerate(keys):
        original = np.load(
            io.BytesIO(await artifacts.read_bytes(storage.Object(key), max_bytes=10**6))
        )
        original_text = np.load(
            io.BytesIO(
                await artifacts.read_bytes(
                    storage.Object(key.replace(".npy", "_text.npy")), max_bytes=10**6
                )
            )
        )
        assert images[row].tobytes() == original.tobytes()
        assert texts[row].tobytes() == original_text.tobytes()
    assert await _positions(index_db) == {key: (0, row) for row, key in enumerate(keys)}


async def test_a_missing_text_vector_becomes_a_zero_row(
    index_db: SavepointDb, run_sync_seed, tmp_path
) -> None:
    artifacts = Memory()
    await _seed_corpus(index_db, run_sync_seed, artifacts, count=2, text=False)
    await _seal(index_db, artifacts, tmp_path, shard_rows=2)

    texts = np.load(
        io.BytesIO(
            await artifacts.read_bytes(
                storage.Object(pack.locate(_MODEL, 0, text=True)), max_bytes=10**6
            )
        )
    )
    assert texts.shape == (2, _DIM)
    assert not texts.any()


async def test_sealing_is_idempotent_and_never_reseals_a_recorded_shard(
    index_db: SavepointDb, run_sync_seed, tmp_path
) -> None:
    artifacts = Memory()
    await _seed_corpus(index_db, run_sync_seed, artifacts, count=4)
    calls = _Calls()

    await _seal(index_db, artifacts, tmp_path, shard_rows=2, calls=calls)
    second = await pack.plan(index_db, model=_MODEL, shard_rows=2)
    again = await _seal(index_db, artifacts, tmp_path, shard_rows=2, calls=calls)

    assert second.shards == [] and second.unsealed == 0
    assert (again.shards, again.rows) == (0, 0)
    assert calls.calls == 2


async def test_an_interrupted_seal_resumes_without_losing_or_duplicating_rows(
    index_db: SavepointDb, run_sync_seed, tmp_path
) -> None:
    artifacts = _FailingUpload()
    keys = await _seed_corpus(index_db, run_sync_seed, artifacts, count=4)

    with pytest.raises(pack.Failed, match="Unavailable"):
        await _seal(index_db, artifacts, tmp_path, shard_rows=2)

    assert set((await _positions(index_db)).values()) == {(None, None)}

    healthy = Memory()
    healthy._objects = artifacts._objects  # noqa: SLF001
    resumed = await _seal(index_db, healthy, tmp_path, shard_rows=2)

    assert resumed.rows == 4
    assert sorted((await _positions(index_db)).items()) == [
        (keys[0], (0, 0)),
        (keys[1], (0, 1)),
        (keys[2], (1, 0)),
        (keys[3], (1, 1)),
    ]


async def test_a_seal_that_fails_partway_keeps_the_shards_it_finished(
    index_db: SavepointDb, run_sync_seed, tmp_path
) -> None:
    artifacts = _FailingSecondShard()
    keys = await _seed_corpus(index_db, run_sync_seed, artifacts, count=4)

    with pytest.raises(pack.Failed, match="Unavailable"):
        await _seal(index_db, artifacts, tmp_path, shard_rows=2)

    positions = await _positions(index_db)
    assert positions[keys[0]] == (0, 0)
    assert positions[keys[1]] == (0, 1)
    assert positions[keys[2]] == (None, None)
    assert positions[keys[3]] == (None, None)

    healthy = Memory()
    healthy._objects = artifacts._objects  # noqa: SLF001
    resumed = await _seal(index_db, healthy, tmp_path, shard_rows=2)

    assert resumed.rows == 2
    assert (await _positions(index_db))[keys[2]] == (1, 0)


async def test_a_new_seal_numbers_shards_above_the_recorded_high_water_mark(
    index_db: SavepointDb, run_sync_seed, tmp_path
) -> None:
    artifacts = Memory()
    await _seed_corpus(index_db, run_sync_seed, artifacts, count=2)
    await _seal(index_db, artifacts, tmp_path, shard_rows=2)

    def seed(session: Session) -> None:
        image = create_image(session=session)
        processing = create_processing(session=session, image=image)
        processing.embed_status = ProcessingStatus.DONE
        processing.embed_model = _MODEL
        processing.embed_dim = _DIM
        processing.embed_s3_key = f"{_PREFIX}/api-ingested/later.npy"
        processing.embed_text_present = False
        session.flush()

    await run_sync_seed(seed)
    await artifacts.put_bytes(
        storage.Object(f"{_PREFIX}/api-ingested/later.npy"),
        _npy(_vector(9)),
        content_type="application/octet-stream",
    )

    later = await pack.plan(index_db, model=_MODEL, shard_rows=1)

    assert [shard.number for shard in later.shards] == [1]


async def test_a_sealed_rebuild_matches_the_unsealed_one_it_replaces(
    index_db: SavepointDb, run_sync_seed, tmp_path
) -> None:
    artifacts = Memory()
    await _seed_corpus(index_db, run_sync_seed, artifacts, count=4)
    settings = Settings()

    def seed(session: Session) -> str:
        job = create_job(session=session, type=JobType.REBUILD_INDEX)
        create_search_index_state(session=session, desired_generation=2, active_generation=1)
        return job.id

    job_id = await run_sync_seed(seed)
    loose = await ops.prepare(index_db, settings, _rebuild(job_id))
    assert loose.build is not None
    before = await Gateway(_BuildCalls(), artifacts=artifacts, workspace_dir=tmp_path).build(
        loose.build
    )

    await _seal(index_db, artifacts, tmp_path, shard_rows=2)

    sealed = await ops.prepare(index_db, settings, _rebuild(job_id))
    assert sealed.build is not None
    after = await Gateway(_BuildCalls(), artifacts=artifacts, workspace_dir=tmp_path).build(
        sealed.build
    )

    assert loose.build.planned_reads == 8
    assert sealed.build.planned_reads == 4
    assert before.manifest is not None and after.manifest is not None
    assert (before.manifest.image_count, before.manifest.text_count) == (4, 4)
    assert (after.manifest.image_count, after.manifest.text_count) == (4, 4)
    assert before.manifest.dimension == after.manifest.dimension
    for name in ("index.faiss", "mapping.json"):
        assert _sha(before.manifest, name) == _sha(after.manifest, name)


def _rebuild(job_id: str) -> index.PrepareInput:
    return index.PrepareInput(
        job_id=job_id,
        workflow_id=rule.workflow_id(job_id),
        force=True,
        trigger=index.Trigger.MANUAL,
        model=_MODEL,
        index_type="flat",
    )


def _sha(manifest: index.Manifest, name: str) -> str:
    return next(file.sha256 for file in manifest.files if file.name == name)


async def test_upload_verification_rejects_a_corrupted_shard(
    index_db: SavepointDb, run_sync_seed, tmp_path
) -> None:
    class _Corrupting(Memory):
        async def stat(self, obj):  # noqa: ANN001, ANN202
            info = await super().stat(obj)
            if info is not None and "/shards/" in obj.key:
                return info.model_copy(update={"length": info.length + 1})
            return info

    artifacts = _Corrupting()
    await _seed_corpus(index_db, run_sync_seed, artifacts, count=2)

    with pytest.raises(pack.Failed, match="has length"):
        await _seal(index_db, artifacts, tmp_path, shard_rows=2)

    assert set((await _positions(index_db)).values()) == {(None, None)}
