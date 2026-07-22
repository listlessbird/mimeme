from __future__ import annotations

import asyncio
import socket

import pytest

from mimeme.compute import protocol


def test_encode_prefixes_length() -> None:
    frame = protocol.encode(b"hello")
    assert len(frame) == 4 + 5
    assert frame[4:] == b"hello"


def test_encode_rejects_oversized() -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.encode(b"x" * (64 * 1024 * 1024 + 1))


async def test_async_frame_roundtrip() -> None:
    left, right = socket.socketpair()
    reader, writer = await asyncio.open_connection(sock=left)
    server_reader, server_writer = await asyncio.open_connection(sock=right)

    await protocol.write_frame(writer, b"ping")
    received = await protocol.read_frame(server_reader)
    assert received == b"ping"

    writer.close()
    server_writer.close()


def test_sync_frame_roundtrip() -> None:
    left, right = socket.socketpair()
    protocol.send_frame(left, b"payload")
    assert protocol.recv_frame(right) == b"payload"
    left.close()
    right.close()


def test_recv_frame_eof() -> None:
    left, right = socket.socketpair()
    left.close()
    with pytest.raises(EOFError):
        protocol.recv_frame(right)
    right.close()
