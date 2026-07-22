from __future__ import annotations

import asyncio
from pathlib import Path

from mimeme import storage
from mimeme.compute.jobs import Jobs
from mimeme.compute.model import (
    AnnotateReply,
    AnnotateSpec,
    ChildOk,
    EmbedCall,
    EmbedReply,
    EmbedReplyItem,
    EmbedSpec,
    EmbedSpecItem,
)
from mimeme.compute.supervisor import ChildDead


class InMemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def read_bytes(self, obj: storage.Object, *, max_bytes: int) -> bytes:
        if obj.key not in self.objects:
            raise storage.Missing(obj.key)
        return self.objects[obj.key]

    async def put_bytes(
        self, obj: storage.Object, data: bytes, *, content_type: str
    ) -> storage.Info:
        self.objects[obj.key] = data
        return storage.Info(object=obj, length=len(data))


class FakeSupervisor:
    def __init__(self) -> None:
        self.restarted: list[str] = []
        self.fail_ids: set[int] = set()
        self.dead = False
        self.block = asyncio.Event()
        self.block.set()

    async def call(self, role: str, request: bytes) -> bytes:
        if self.dead:
            raise ChildDead("inference child is not running")
        await self.block.wait()
        if b'"op":"annotate"' in request or b'"annotate"' in request:
            reply = AnnotateReply(caption="a cat", caption_model="m", ocr_text="hi", ocr_model="m")
            return ChildOk(result=reply.model_dump()).model_dump_json().encode()
        call = EmbedCall.model_validate_json(request)
        items = []
        for item in call.items:
            if item.image_id in self.fail_ids:
                items.append(EmbedReplyItem(image_id=item.image_id, ok=False, error="bad image"))
                continue
            Path(item.image_out).write_bytes(b"IMG")
            Path(item.text_out).write_bytes(b"TXT")
            items.append(
                EmbedReplyItem(image_id=item.image_id, ok=True, model="siglip", dimension=768)
            )
        return ChildOk(result=EmbedReply(items=items).model_dump()).model_dump_json().encode()

    async def restart(self, role: str) -> None:
        self.restarted.append(role)


async def _wait(jobs: Jobs, job_id: str, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        state = jobs.get(job_id)
        if state and state.status in ("succeeded", "failed", "cancelled"):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("job did not finish")


def _make_jobs(
    tmp_path: Path, supervisor: FakeSupervisor
) -> tuple[Jobs, InMemoryStore, InMemoryStore]:
    media = InMemoryStore()
    artifacts = InMemoryStore()
    jobs = Jobs(
        supervisor,  # type: ignore[arg-type]
        media=media,  # type: ignore[arg-type]
        artifacts=artifacts,  # type: ignore[arg-type]
        workspace_dir=tmp_path / "work",
    )
    return jobs, media, artifacts


async def test_annotate_job_happy(tmp_path: Path) -> None:
    supervisor = FakeSupervisor()
    jobs, media, _ = _make_jobs(tmp_path, supervisor)
    media.objects["images/a.jpg"] = b"bytes"
    jobs.submit("j1", AnnotateSpec(media_key="images/a.jpg", length="normal"))
    await _wait(jobs, "j1")
    state = jobs.get("j1")
    assert state is not None and state.status == "succeeded"
    assert state.result == {
        "caption": "a cat",
        "caption_model": "m",
        "ocr_text": "hi",
        "ocr_model": "m",
    }


async def test_embed_job_uploads_and_preserves_order(tmp_path: Path) -> None:
    supervisor = FakeSupervisor()
    jobs, media, artifacts = _make_jobs(tmp_path, supervisor)
    media.objects["images/1.jpg"] = b"1"
    media.objects["images/2.jpg"] = b"2"
    spec = EmbedSpec(
        model="siglip",
        items=[
            EmbedSpecItem(
                image_id=1,
                media_key="images/1.jpg",
                text="t1",
                sha256="s1",
                image_key="e/1.npy",
                text_key="e/1_text.npy",
            ),
            EmbedSpecItem(
                image_id=2,
                media_key="images/2.jpg",
                text="t2",
                sha256="s2",
                image_key="e/2.npy",
                text_key="e/2_text.npy",
            ),
        ],
    )
    jobs.submit("e1", spec)
    await _wait(jobs, "e1")
    state = jobs.get("e1")
    assert state is not None and state.status == "succeeded"
    assert state.result is not None
    ids = [item["image_id"] for item in state.result["items"]]
    assert ids == [1, 2]
    assert artifacts.objects["e/1.npy"] == b"IMG"
    assert artifacts.objects["e/2_text.npy"] == b"TXT"


async def test_embed_partial_failure(tmp_path: Path) -> None:
    supervisor = FakeSupervisor()
    supervisor.fail_ids = {2}
    jobs, media, artifacts = _make_jobs(tmp_path, supervisor)
    media.objects["images/1.jpg"] = b"1"
    media.objects["images/2.jpg"] = b"2"
    spec = EmbedSpec(
        model="siglip",
        items=[
            EmbedSpecItem(
                image_id=1,
                media_key="images/1.jpg",
                text="t",
                sha256="s1",
                image_key="e/1.npy",
                text_key="e/1_text.npy",
            ),
            EmbedSpecItem(
                image_id=2,
                media_key="images/2.jpg",
                text="t",
                sha256="s2",
                image_key="e/2.npy",
                text_key="e/2_text.npy",
            ),
        ],
    )
    jobs.submit("e2", spec)
    await _wait(jobs, "e2")
    state = jobs.get("e2")
    assert state is not None and state.result is not None
    items = {item["image_id"]: item for item in state.result["items"]}
    assert items[1]["ok"] is True
    assert items[2]["ok"] is False
    assert "e/2.npy" not in artifacts.objects


async def test_idempotent_resubmit(tmp_path: Path) -> None:
    supervisor = FakeSupervisor()
    supervisor.block.clear()
    jobs, media, _ = _make_jobs(tmp_path, supervisor)
    media.objects["images/a.jpg"] = b"bytes"
    first = jobs.submit("dup", AnnotateSpec(media_key="images/a.jpg"))
    second = jobs.submit("dup", AnnotateSpec(media_key="images/a.jpg"))
    assert first.job_id == second.job_id
    assert len(jobs._jobs) == 1  # noqa: SLF001
    supervisor.block.set()
    await _wait(jobs, "dup")


async def test_cancel_marks_cancelled_and_restarts_child(tmp_path: Path) -> None:
    supervisor = FakeSupervisor()
    supervisor.block.clear()
    jobs, media, _ = _make_jobs(tmp_path, supervisor)
    media.objects["images/a.jpg"] = b"bytes"
    jobs.submit("c1", AnnotateSpec(media_key="images/a.jpg"))
    await asyncio.sleep(0.05)
    state = await jobs.cancel("c1")
    assert state is not None and state.status == "cancelled"
    assert supervisor.restarted == ["inference"]


async def test_child_dead_fails_job(tmp_path: Path) -> None:
    supervisor = FakeSupervisor()
    supervisor.dead = True
    jobs, media, _ = _make_jobs(tmp_path, supervisor)
    media.objects["images/a.jpg"] = b"bytes"
    jobs.submit("d1", AnnotateSpec(media_key="images/a.jpg"))
    await _wait(jobs, "d1")
    state = jobs.get("d1")
    assert state is not None and state.status == "failed"
    assert "child_dead" in (state.error or "")
