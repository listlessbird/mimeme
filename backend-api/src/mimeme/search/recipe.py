"""Closed, versioned retrieval recipes and legacy name resolution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mimeme.search.error import Invalid, Unavailable

RecipeId = Literal["image_only", "image_siglip_text"]
RecipeName = Literal["image", "hybrid", "image_only", "image_siglip_text"]
RetrieverId = Literal["siglip_image", "siglip_text"]


class UnknownRecipe(Invalid):
    pass


class UnavailableRetriever(Unavailable):
    pass


class Definition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: RecipeId
    version: Literal[1] = 1
    label: str = Field(min_length=1)
    retrievers: tuple[RetrieverId, ...] = Field(min_length=1)
    candidate_depth: int = Field(ge=1, le=1000)
    rrf_k: int = Field(ge=1)


_ALIASES: dict[RecipeName, RecipeId] = {
    "image": "image_only",
    "hybrid": "image_siglip_text",
    "image_only": "image_only",
    "image_siglip_text": "image_siglip_text",
}

_DEFINITIONS: dict[RecipeId, Definition] = {
    "image_only": Definition(
        id="image_only",
        label="Image only",
        retrievers=("siglip_image",),
        candidate_depth=1000,
        rrf_k=60,
    ),
    "image_siglip_text": Definition(
        id="image_siglip_text",
        label="Image and SigLIP text",
        retrievers=("siglip_image", "siglip_text"),
        candidate_depth=1000,
        rrf_k=60,
    ),
}


def id_of(name: str) -> RecipeId:
    try:
        return _ALIASES[name]  # type: ignore[index]
    except KeyError as exc:
        raise UnknownRecipe(f"unknown search recipe: {name}") from exc


def resolve(
    name: str,
    *,
    available: set[RetrieverId] | None = None,
) -> Definition:
    definition = _DEFINITIONS[id_of(name)]
    if available is not None:
        missing = set(definition.retrievers).difference(available)
        if missing:
            names = ", ".join(sorted(missing))
            raise UnavailableRetriever(
                f"recipe {definition.id!r} requires unavailable retrievers: {names}"
            )
    return definition


def all() -> tuple[Definition, ...]:
    return tuple(_DEFINITIONS[recipe_id] for recipe_id in _DEFINITIONS)
