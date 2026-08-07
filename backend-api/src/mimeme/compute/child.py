from __future__ import annotations

import os
import socket
from pathlib import Path

from pydantic import TypeAdapter

from mimeme.compute import image as image_role
from mimeme.compute.model import (
    AnnotateCall,
    ChildErr,
    ChildOk,
    EmbedCall,
    InferenceCall,
    InspectCall,
    Role,
)
from mimeme.compute.protocol import recv_frame, send_frame
from mimeme.index.model import BuildCall, IndexCall

_INFERENCE_CALL = TypeAdapter(InferenceCall)
_INDEX_CALL = TypeAdapter(IndexCall)


def run_child(role: Role, socket_path: str) -> None:
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    handler = _build_handler(role)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    os.chmod(socket_path, 0o600)
    try:
        while True:
            conn, _ = server.accept()
            try:
                _serve_once(conn, handler)
            finally:
                conn.close()
    finally:
        server.close()
        path.unlink(missing_ok=True)


def _serve_once(conn: socket.socket, handler) -> None:  # noqa: ANN001
    try:
        raw = recv_frame(conn)
    except EOFError:
        return
    try:
        result = handler(raw)
        response = ChildOk(result=result)
    except Exception as exc:
        response = ChildErr(
            error=f"{type(exc).__name__}: {exc}",
            code=getattr(exc, "code", None),
        )
    send_frame(conn, response.model_dump_json().encode("utf-8"))


def _build_handler(role: Role):  # noqa: ANN202
    if role == "image":

        def handle_image(raw: bytes) -> dict:
            call = InspectCall.model_validate_json(raw)
            return image_role.inspect(call).model_dump()

        return handle_image

    if role == "inference":
        from mimeme.compute.inference import Models
        from mimeme.config import Settings

        models = Models(Settings().inference)

        def handle_inference(raw: bytes) -> dict:
            call = _INFERENCE_CALL.validate_json(raw)
            if isinstance(call, AnnotateCall):
                return models.annotate(call).model_dump()
            if isinstance(call, EmbedCall):
                return models.embed(call).model_dump()
            raise ValueError("unknown inference call")

        return handle_inference

    if role == "search":
        from mimeme.compute.search import Resident, dispatch

        resident = Resident()

        def handle_search(raw: bytes) -> dict:
            return dispatch(resident, raw)

        return handle_search

    if role == "index":
        from mimeme.compute.index import build, pack

        def handle_index(raw: bytes) -> dict:
            call = _INDEX_CALL.validate_json(raw)
            if isinstance(call, BuildCall):
                return build(call.build).model_dump()
            return pack(call).model_dump()

        return handle_index

    raise ValueError(f"role {role} is not enabled")
