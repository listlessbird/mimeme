from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["image", "inference", "search", "index"]

ENABLED_ROLES: tuple[Role, ...] = ("image", "inference")
RESERVED_ROLES: tuple[Role, ...] = ("search", "index")

StorageRole = Literal["media", "artifacts"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class InspectCall(_Frozen):
    op: Literal["inspect"] = "inspect"
    path: str


class ImageInfo(_Frozen):
    format: str
    mode: str
    width: int
    height: int
    sha256: str
    phash: str


class AnnotateCall(_Frozen):
    op: Literal["annotate"] = "annotate"
    path: str
    length: str = "normal"


class AnnotateReply(_Frozen):
    caption: str
    caption_model: str
    ocr_text: str
    ocr_model: str


class EmbedCallItem(_Frozen):
    image_id: int
    path: str
    text: str
    image_out: str
    text_out: str


class EmbedCall(_Frozen):
    op: Literal["embed"] = "embed"
    items: list[EmbedCallItem]


class EmbedReplyItem(_Frozen):
    image_id: int
    ok: bool
    model: str | None = None
    dimension: int | None = None
    error: str | None = None


class EmbedReply(_Frozen):
    items: list[EmbedReplyItem]


ImageCall = InspectCall
InferenceCall = Annotated[AnnotateCall | EmbedCall, Field(discriminator="op")]


class ChildOk(_Frozen):
    ok: Literal[True] = True
    result: dict


class ChildErr(_Frozen):
    ok: Literal[False] = False
    error: str


ChildResponse = Annotated[ChildOk | ChildErr, Field(discriminator="ok")]


class AnnotateSpec(_Frozen):
    op: Literal["annotate"] = "annotate"
    media_key: str
    length: str = "normal"


class EmbedSpecItem(_Frozen):
    image_id: int
    media_key: str
    text: str
    sha256: str
    image_key: str
    text_key: str


class EmbedSpec(_Frozen):
    op: Literal["embed"] = "embed"
    model: str
    items: list[EmbedSpecItem]


JobSpec = Annotated[AnnotateSpec | EmbedSpec, Field(discriminator="op")]


class AnnotateResult(_Frozen):
    caption: str
    caption_model: str
    ocr_text: str
    ocr_model: str


class EmbedResultItem(_Frozen):
    image_id: int
    ok: bool
    image_key: str | None = None
    text_key: str | None = None
    model: str | None = None
    dimension: int | None = None
    error: str | None = None


class EmbedResult(_Frozen):
    items: list[EmbedResultItem]


Status = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class JobState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Status
    phase: str | None = None
    progress: float = 0.0
    result: dict | None = None
    error: str | None = None


class RoleStatus(_Frozen):
    role: Role
    state: Literal["ready", "failed", "disabled", "starting"]
    detail: str | None = None


class Readiness(_Frozen):
    ok: bool
    roles: list[RoleStatus]
