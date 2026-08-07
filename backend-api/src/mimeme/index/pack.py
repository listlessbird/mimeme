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
    text_present: bool


class Shard(_Frozen):
    number: int = Field(ge=0)
    model: str = Field(min_length=1)
    members: list[Member]

    @property
    def image_key(self) -> str:
        return locate(self.model, self.number, text=False)

    @property
    def text_key(self) -> str:
        return locate(self.model, self.number, text=True)


class Plan(_Frozen):
    model: str = Field(min_length=1)
    shard_rows: int = Field(gt=0)
    unsealed: int = Field(ge=0)
    shards: list[Shard]

    @property
    def tail(self) -> int:
        return self.unsealed - sum(len(shard.members) for shard in self.shards)


def locate(model: str, number: int, *, text: bool) -> str:
    family = "text" if text else "image"
    return f"{inference.embedding_prefix(model)}shards/{family}/{number:06d}.npy"


def reads(embeddings: list[Embedding]) -> int:
    image_shards = {item.shard for item in embeddings if item.sealed}
    text_shards = {item.shard for item in embeddings if item.sealed and item.text_present}
    loose = sum(1 + int(item.text_key is not None) for item in embeddings if not item.sealed)
    return len(image_shards) + len(text_shards) + loose


async def plan(db: Db, *, model: str, shard_rows: int, max_shards: int | None = None) -> Plan:
    limit = None if max_shards is None else shard_rows * max_shards
    async with db.read_session() as session:
        store = Store(session)
        rows = await store.unsealed(model=model, limit=limit)
        first = await store.next_shard(model=model)
    members = [
        Member(
            image_id=row.image_id,
            image_key=str(row.embed_s3_key),
            text_present=bool(row.embed_text_present),
        )
        for row in rows
    ]
    shards = [
        Shard(
            number=first + position,
            model=model,
            members=members[position * shard_rows : (position + 1) * shard_rows],
        )
        for position in range(len(members) // shard_rows)
    ]
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
                image_key=shard.image_key,
                text_key=shard.text_key,
                members=[
                    SealMember(
                        image_id=member.image_id,
                        image_key=member.image_key,
                        text_key=(
                            inference.text_embedding_key(member.image_key)
                            if member.text_present
                            else None
                        ),
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
) -> Sealed:
    async with _exclusive(db, model):
        target = await plan(db, model=model, shard_rows=shard_rows, max_shards=max_shards)
        if not target.shards:
            return Sealed(model=model, shards=0, rows=0)
        result = await packer.seal(request(target, job_id=job_id))
        planned = {shard.number: shard for shard in target.shards}
        rows = 0
        for done in result.shards:
            shard = planned[done.number]
            if done.rows != len(shard.members):
                raise Failed(
                    f"shard {done.number} sealed {done.rows} rows for {len(shard.members)} members"
                )
            async with db.write_session() as session:
                await Store(session).record_shard(
                    shard=shard.number,
                    image_ids=[member.image_id for member in shard.members],
                )
            rows += done.rows
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
        done.append(SealedShard(number=shard.number, rows=rows))
    return SealResult(shards=done)


async def _seal_one(
    artifacts: storage.Store,
    calls: Calls,
    workspace_dir: Path,
    model: str,
    shard: SealShard,
) -> int:
    name = model.replace("/", "_")
    workspace = Workspace.create(workspace_dir, f"seal-{name}-{shard.number:06d}")
    try:
        members: list[LocalMember] = []
        for position, member in enumerate(shard.members):
            image_path = workspace.path(f"image-{position}.npy")
            await _download(artifacts, member.image_key, image_path)
            text_path: Path | None = None
            if member.text_key is not None:
                text_path = workspace.path(f"text-{position}.npy")
                await _download(artifacts, member.text_key, text_path)
            members.append(
                LocalMember(
                    image_id=member.image_id,
                    image_path=str(image_path),
                    text_path=str(text_path) if text_path else None,
                )
            )
        packed = await _pack(
            calls,
            members,
            image_out=workspace.path("shard-image.npy"),
            text_out=workspace.path("shard-text.npy"),
        )
        if packed.rows != len(shard.members):
            raise Failed(
                f"shard {shard.number} packed {packed.rows} rows for {len(shard.members)} members"
            )
        await _upload(artifacts, packed.image, shard.image_key)
        await _upload(artifacts, packed.text, shard.text_key)
        return packed.rows
    finally:
        workspace.close()


async def _pack(
    calls: Calls, members: list[LocalMember], *, image_out: Path, text_out: Path
) -> Packed:
    raw = await calls.call(
        "index",
        PackCall(members=members, image_out=str(image_out), text_out=str(text_out))
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
