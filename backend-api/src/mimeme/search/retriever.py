from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from mimeme.search.recipe import RetrieverId

ContextT = TypeVar("ContextT", contravariant=True)


@dataclass(frozen=True)
class Scored:
    image_id: int
    score: float


class Retriever(Protocol[ContextT]):
    id: RetrieverId

    def search(self, context: ContextT, *, depth: int) -> list[Scored]: ...
