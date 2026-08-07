from __future__ import annotations

import asyncio
import json
import socket
import struct

from pydantic import ValidationError

from mimeme.compute.model import ChildErr, ChildOk

_HEADER = struct.Struct(">I")
_MAX_FRAME = 64 * 1024 * 1024


class ProtocolError(Exception):
    pass


def parse_reply(raw: bytes) -> ChildOk | ChildErr:
    try:
        payload = json.loads(raw)
        return (
            ChildOk.model_validate(payload)
            if payload.get("ok")
            else ChildErr.model_validate(payload)
        )
    except (ValueError, ValidationError) as exc:
        raise ProtocolError(f"invalid compute child response: {exc}") from exc


def encode(payload: bytes) -> bytes:
    if len(payload) > _MAX_FRAME:
        raise ProtocolError(f"frame exceeds {_MAX_FRAME} bytes")
    return _HEADER.pack(len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    header = await reader.readexactly(_HEADER.size)
    (length,) = _HEADER.unpack(header)
    if length > _MAX_FRAME:
        raise ProtocolError(f"frame length {length} exceeds {_MAX_FRAME}")
    return await reader.readexactly(length)


async def write_frame(writer: asyncio.StreamWriter, payload: bytes) -> None:
    writer.write(encode(payload))
    await writer.drain()


def recv_frame(conn: socket.socket) -> bytes:
    header = _recv_exact(conn, _HEADER.size)
    if header is None:
        raise EOFError
    (length,) = _HEADER.unpack(header)
    if length > _MAX_FRAME:
        raise ProtocolError(f"frame length {length} exceeds {_MAX_FRAME}")
    body = _recv_exact(conn, length)
    if body is None:
        raise EOFError
    return body


def send_frame(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(encode(payload))


def _recv_exact(conn: socket.socket, size: int) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = conn.recv(size - len(buffer))
        if not chunk:
            return None
        buffer += chunk
    return bytes(buffer)
