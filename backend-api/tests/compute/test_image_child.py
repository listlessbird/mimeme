from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from mimeme.compute.image import inspect
from mimeme.compute.model import ChildOk, ImageInfo, InspectCall
from mimeme.compute.supervisor import ChildDead, Supervisor
from mimeme.search import Status


def _png(path: Path, size: tuple[int, int] = (32, 24)) -> Path:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 50, 25)).save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())
    return path


def test_inspect_direct(tmp_path: Path) -> None:
    path = _png(tmp_path / "img.png")
    info = inspect(InspectCall(path=str(path)))
    assert isinstance(info, ImageInfo)
    assert info.format == "PNG"
    assert (info.width, info.height) == (32, 24)
    assert len(info.sha256) == 64
    assert info.phash


async def test_spawned_image_child_roundtrip(tmp_path: Path) -> None:
    supervisor = Supervisor(tmp_path / "sock")
    await supervisor.start()
    try:
        readiness = supervisor.readiness()
        states = {r.role: r.state for r in readiness.roles}
        assert states["image"] == "ready"
        assert states["search"] == "ready"
        assert states["index"] == "ready"

        search_raw = await supervisor.call("search", b'{"op":"search.status"}')
        search_response = ChildOk.model_validate_json(search_raw)
        search_status = Status.model_validate(search_response.result)
        assert search_status.ready is False
        assert search_status.serving_version is None

        path = _png(tmp_path / "img.png")
        call = InspectCall(path=str(path))
        raw = await supervisor.call("image", call.model_dump_json().encode("utf-8"))
        response = ChildOk.model_validate_json(raw)
        info = ImageInfo.model_validate(response.result)
        assert (info.width, info.height) == (32, 24)

        await supervisor.restart("image")
        raw2 = await supervisor.call("image", call.model_dump_json().encode("utf-8"))
        assert ChildOk.model_validate_json(raw2)
    finally:
        await supervisor.close()


async def test_child_death_gives_typed_failure(tmp_path: Path) -> None:
    supervisor = Supervisor(tmp_path / "sock")
    await supervisor.start()
    try:
        child = supervisor._children["image"]  # noqa: SLF001
        assert child.process is not None
        child.process.kill()
        child.process.join(2.0)
        with pytest.raises(ChildDead):
            await supervisor.call("image", InspectCall(path="/nope").model_dump_json().encode())
    finally:
        await supervisor.close()
