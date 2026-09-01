from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROJECTION_VERSION = 1


@dataclass(frozen=True)
class SourceFacts:
    item_title: str | None = None
    title: str | None = None
    tags: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    origin: str | None = None
    year: str | None = None
    description: str | None = None


class SearchDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    projection_version: Literal[1] = PROJECTION_VERSION
    image_id: int = Field(gt=0)
    titles: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    years: tuple[str, ...] = ()
    captions: tuple[str, ...] = ()
    ocr_texts: tuple[str, ...] = ()
    descriptions: tuple[str, ...] = ()


def source_facts(item_title: str | None, known_facts: Mapping[str, object]) -> SourceFacts:
    return SourceFacts(
        item_title=item_title,
        title=_text(known_facts.get("title")),
        tags=_text_list(known_facts.get("tags")),
        categories=_text_list(known_facts.get("categories")),
        types=_text_list(known_facts.get("types")),
        origin=_text(known_facts.get("origin")),
        year=_text(known_facts.get("year")),
        description=_text(known_facts.get("description")),
    )


def project(
    image_id: int,
    *,
    sources: Iterable[SourceFacts] = (),
    caption: str | None = None,
    ocr_text: str | None = None,
) -> SearchDocument:
    source_values = tuple(sources)
    return SearchDocument(
        image_id=image_id,
        titles=_values(
            value for source in source_values for value in (source.item_title, source.title)
        ),
        tags=_values(value for source in source_values for value in source.tags),
        categories=_values(value for source in source_values for value in source.categories),
        types=_values(value for source in source_values for value in source.types),
        origins=_values(source.origin for source in source_values),
        years=_values(source.year for source in source_values),
        captions=_values((caption,)),
        ocr_texts=_values((ocr_text,)),
        descriptions=_values(source.description for source in source_values),
    )


def canonical_json(value: SearchDocument) -> str:
    return value.model_dump_json()


def _values(values: Iterable[str | None]) -> tuple[str, ...]:
    normalized = {_normalize(value) for value in values}
    present = (value for value in normalized if value is not None)
    return tuple(sorted(present, key=lambda value: (value.casefold(), value)))


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _text_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))
