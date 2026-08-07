from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Input(_Frozen):
    image_id: int
    media_key: str
    length: str = "normal"


class Annotation(_Frozen):
    image_id: int
    caption: str
    caption_model: str
    ocr_text: str
    ocr_model: str


class Item(_Frozen):
    image_id: int
    media_key: str
    text: str = ""
    sha256: str
    dataset: str | None = None


class Batch(_Frozen):
    items: list[Item]
    dataset: str | None = None


class Embedding(_Frozen):
    image_id: int
    image_embedding_key: str
    text_embedding_key: str
    model: str
    dimension: int


class Ok(_Frozen):
    kind: Literal["ok"] = "ok"
    embedding: Embedding


class Failed(_Frozen):
    kind: Literal["failed"] = "failed"
    image_id: int
    error: str


BatchItem = Annotated[Ok | Failed, Field(discriminator="kind")]


class BatchResult(_Frozen):
    items: list[BatchItem]

    @property
    def results(self) -> list[Embedding]:
        return [item.embedding for item in self.items if isinstance(item, Ok)]

    @property
    def failed_ids(self) -> list[int]:
        return [item.image_id for item in self.items if isinstance(item, Failed)]


class Error(Exception):
    pass


class Invalid(Error):
    pass


class Unavailable(Error):
    pass


class Timeout(Error):
    pass


_TEXT_SUFFIX = "_text.npy"


def embedding_prefix(model: str) -> str:
    return f"embeddings/{model.replace('/', '_')}/"


def image_embedding_key(*, sha256: str, model: str, dataset: str | None) -> str:
    return f"{embedding_prefix(model)}{dataset or 'api-ingested'}/{sha256}.npy"


def text_embedding_key(image_key: str) -> str:
    return image_key.replace(".npy", _TEXT_SUFFIX)


def is_text_embedding_key(key: str) -> bool:
    return key.endswith(_TEXT_SUFFIX)


def image_embedding_key_of(text_key: str) -> str:
    if not is_text_embedding_key(text_key):
        raise ValueError(f"not a text embedding key: {text_key}")
    return text_key.removesuffix(_TEXT_SUFFIX) + ".npy"
