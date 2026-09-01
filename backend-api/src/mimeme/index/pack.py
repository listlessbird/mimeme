from __future__ import annotations

import zlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import anyio
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from mimeme import inference, storage
from mimeme.compute.model import ChildErr, Role
from mimeme.compute.protocol import parse_reply
from mimeme.compute.workspace import Workspace
from mimeme.db import Db
from mimeme.index.model import (
    Embedding,
    LocalMember,
    PackCall,
    Packed,
    PackedFile,
    Seal,
    Sealed,
    SealedShard,
    SealMember,
    SealResult,
    SealShard,
)
from mimeme.index.store import Store

_UPLOAD_CHUNK = 1024 * 1024
_LOCK_CLASS = 0x6D696D65


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Failed(Exception):
    pass


class Busy(Exception):
    pass


class Calls(Protocol):
    async def call(self, role: Role, request: bytes) -> bytes: ...


class Packer(Protocol):
    async def seal(self, request: Seal) -> SealResult: ...


class Member(_Frozen):
    image_id: int = Field(gt=0)
    image_key: str = Field(min_length=1)


class Shard(_Frozen):
    number: int = Field(ge=0)
    seq: int = Field(ge=0)
    model: str = Field(min_length=1)
    first_row: int = Field(ge=0)
    base_rows: int = Field(ge=0)
    sealed: bool
    members: list[Member]

    @property
    def image_key(self) -> str:
        return locate(self.model, self.number, self.seq)

    @property
    def base_image_key(self) -> str | None:
        if self.base_rows == 0:
            return None
        return locate(self.model, self.number, self.seq - 1)


class Plan(_Frozen):
    model: str = Field(min_length=1)
    shard_rows: int = Field(gt=0)
    unsealed: int = Field(ge=0)
    shards: list[Shard]

    @property
    def absorbed(self) -> int:
        return sum(len(shard.members) for shard in self.shards)

    @property
    def tail(self) -> int:
        return self.unsealed - self.absorbed


def locate(model: str, number: int, seq: int) -> str:
    return f"{inference.embedding_prefix(model)}shards/image/{number:06d}-{seq:04d}.npy"


def reads(embeddings: list[Embedding]) -> int:
    image_shards = {item.shard for item in embeddings if item.sealed}
    loose = sum(1 for item in embeddings if not item.sealed)
    return len(image_shards) + loose


async def plan(
    db: Db,
    *,
    model: str,
    shard_rows: int,
    max_shards: int | None = None,
    min_rows: int = 1,
) -> Plan:
    limit = None if max_shards is None else shard_rows * max_shards
    async with db.read_session() as session:
        store = Store(session)
        rows = await store.unsealed(model=model, limit=limit)
        open_shard = await store.open_shard(model=model)
        next_number = await store.next_shard(model=model)
    members = [
        Member(
            image_id=row.image_id,
            image_key=str(row.embed_s3_key),
        )
        for row in rows
    ]
    empty = Plan(model=model, shard_rows=shard_rows, unsealed=len(members), shards=[])
    if not members:
        return empty

    shards: list[Shard] = []
    remaining = members
    if open_shard is not None and open_shard.row_count < shard_rows:
        take = min(shard_rows - open_shard.row_count, len(remaining))
        shards.append(
            Shard(
                number=open_shard.number,
                seq=open_shard.seq + 1,
                model=model,
                first_row=open_shard.row_count,
                base_rows=open_shard.row_count,
                sealed=open_shard.row_count + take == shard_rows,
                members=remaining[:take],
            )
        )
        remaining = remaining[take:]
    while remaining:
        take = min(shard_rows, len(remaining))
        shards.append(
            Shard(
                number=next_number,
                seq=0,
                model=model,
                first_row=0,
                base_rows=0,
                sealed=take == shard_rows,
                members=remaining[:take],
            )
        )
        next_number += 1
        remaining = remaining[take:]

    absorbed = sum(len(shard.members) for shard in shards)
    if absorbed < min_rows and not any(shard.sealed for shard in shards):
        return empty
    return Plan(model=model, shard_rows=shard_rows, unsealed=len(members), shards=shards)


def _lock_key(model: str) -> int:
    digest = zlib.crc32(model.encode())
    return digest - 2**32 if digest >= 2**31 else digest


@asynccontextmanager
async def _exclusive(db: Db, model: str) -> AsyncIterator[None]:
    key = _lock_key(model)
    async with db.write_session() as session:
        acquired = await session.scalar(select(func.pg_try_advisory_xact_lock(_LOCK_CLASS, key)))
        if not acquired:
            raise Busy(f"another seal already holds the pack lock for {model}")
        yield


def request(target: Plan, *, job_id: str) -> Seal:
    return Seal(
        job_id=job_id,
        model=target.model,
        shards=[
            SealShard(
                number=shard.number,
                seq=shard.seq,
                image_key=shard.image_key,
                base_image_key=shard.base_image_key,
                base_rows=shard.base_rows,
                sealed=shard.sealed,
                members=[
                    SealMember(
                        image_id=member.image_id,
                        image_key=member.image_key,
                    )
                    for member in shard.members
                ],
            )
            for shard in target.shards
        ],
    )


async def seal(
    db: Db,
    packer: Packer,
    *,
    job_id: str,
    model: str,
    shard_rows: int,
    max_shards: int | None = None,
    min_rows: int = 1,
) -> Sealed:
    async with _exclusive(db, model):
        target = await plan(
            db,
            model=model,
            shard_rows=shard_rows,
            max_shards=max_shards,
            min_rows=min_rows,
        )
        if not target.shards:
            return Sealed(model=model, shards=0, rows=0)
        result = await packer.seal(request(target, job_id=job_id))
        planned = {shard.number: shard for shard in target.shards}
        rows = 0
        for done in result.shards:
            shard = planned[done.number]
            expected = shard.base_rows + len(shard.members)
            if done.rows != expected or done.seq != shard.seq:
                raise Failed(
                    f"shard {done.number} came back as {done.rows} rows at generation "
                    f"{done.seq}, expected {expected} at {shard.seq}"
                )
            async with db.write_session() as session:
                await Store(session).record_shard(
                    model=model,
                    shard=shard.number,
                    seq=shard.seq,
                    first_row=shard.first_row,
                    image_ids=[member.image_id for member in shard.members],
                    sealed=shard.sealed,
                )
            rows += len(shard.members)
        sealed = Sealed(model=target.model, shards=len(result.shards), rows=rows)
    if result.error is not None:
        raise Failed(result.error)
    return sealed


async def perform(
    artifacts: storage.Store,
    calls: Calls,
    *,
    workspace_dir: Path,
    target: Seal,
) -> SealResult:
    done: list[SealedShard] = []
    for shard in target.shards:
        try:
            rows = await _seal_one(artifacts, calls, workspace_dir, target.model, shard)
        except Exception as failure:
            return SealResult(shards=done, error=f"{type(failure).__name__}: {failure}")
        done.append(SealedShard(number=shard.number, seq=shard.seq, rows=rows))
    return SealResult(shards=done)


async def _seal_one(
    artifacts: storage.Store,
    calls: Calls,
    workspace_dir: Path,
    model: str,
    shard: SealShard,
) -> int:
    name = model.replace("/", "_")
    workspace = Workspace.create(workspace_dir, f"seal-{name}-{shard.number:06d}-{shard.seq:04d}")
    try:
        base_image: Path | None = None
        if shard.base_image_key is not None:
            base_image = workspace.path("base-image.npy")
            await _download(artifacts, shard.base_image_key, base_image)
        members: list[LocalMember] = []
        for position, member in enumerate(shard.members):
            image_path = workspace.path(f"image-{position}.npy")
            await _download(artifacts, member.image_key, image_path)
            members.append(
                LocalMember(
                    image_id=member.image_id,
                    image_path=str(image_path),
                )
            )
        packed = await _pack(
            calls,
            members,
            image_out=workspace.path("shard-image.npy"),
            base_image=base_image,
            base_rows=shard.base_rows,
        )
        expected = shard.base_rows + len(shard.members)
        if packed.rows != expected:
            raise Failed(f"shard {shard.number} packed {packed.rows} rows, expected {expected}")
        await _upload(artifacts, packed.image, shard.image_key)
        return packed.rows
    finally:
        workspace.close()


async def _pack(
    calls: Calls,
    members: list[LocalMember],
    *,
    image_out: Path,
    base_image: Path | None = None,
    base_rows: int = 0,
) -> Packed:
    raw = await calls.call(
        "index",
        PackCall(
            members=members,
            image_out=str(image_out),
            base_image=str(base_image) if base_image else None,
            base_rows=base_rows,
        )
        .model_dump_json()
        .encode(),
    )
    reply = parse_reply(raw)
    if isinstance(reply, ChildErr):
        raise Failed(reply.error)
    return Packed.model_validate(reply.result)


async def _download(artifacts: storage.Store, key: str, target: Path) -> None:
    async with artifacts.read(storage.Object(key)) as chunks:
        async with await anyio.open_file(target, "wb") as handle:
            async for chunk in chunks:
                await handle.write(chunk)


async def _upload(artifacts: storage.Store, file: PackedFile, key: str) -> None:
    async def chunks():  # noqa: ANN202
        async with await anyio.open_file(file.path, "rb") as handle:
            while chunk := await handle.read(_UPLOAD_CHUNK):
                yield chunk

    await artifacts.put(
        storage.Object(key),
        chunks(),
        length=file.length,
        content_type="application/octet-stream",
        checksum=storage.Checksum(value=file.sha256),
    )
    info = await artifacts.stat(storage.Object(key))
    if info is None:
        raise Failed(f"shard object is missing after upload: {key}")
    if info.length != file.length:
        raise Failed(f"shard object has length {info.length}, expected {file.length}: {key}")
    if info.checksum is not None and info.checksum.value != file.sha256:
        raise Failed(f"shard object checksum mismatch: {key}")
