from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mimeme.search.error import Invalid, Unavailable

RecipeId = Literal[
    "image_only",
    "image_siglip_text",
    "image_bm25",
    "image_bge",
    "image_bm25_bge",
]
RecipeName = Literal[
    "image",
    "hybrid",
    "image_only",
    "image_siglip_text",
    "image_bm25",
    "image_bge",
    "image_bm25_bge",
]
RetrieverId = Literal["siglip_image", "siglip_text", "bm25", "bge"]


class UnknownRecipe(Invalid):
    pass


class UnavailableRetriever(Unavailable):
    pass


class Bm25Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    projection_version: Literal[1] = 1
    tokenizer: Literal["porter unicode61"] = "porter unicode61"
    weights: tuple[float, float, float, float, float, float, float]


class Definition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: RecipeId
    version: Literal[1] = 1
    label: str = Field(min_length=1)
    retrievers: tuple[RetrieverId, ...] = Field(min_length=1)
    candidate_depth: int = Field(ge=1, le=1000)
    rrf_k: int = Field(ge=1)
    bm25: Bm25Settings | None = None


_ALIASES: dict[str, RecipeId] = {
    "image": "image_only",
    "hybrid": "image_siglip_text",
    "image_only": "image_only",
    "image_siglip_text": "image_siglip_text",
    "image_bm25": "image_bm25",
    "image_bge": "image_bge",
    "image_bm25_bge": "image_bm25_bge",
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
    "image_bm25": Definition(
        id="image_bm25",
        label="Image and BM25",
        retrievers=("siglip_image", "bm25"),
        candidate_depth=1000,
        rrf_k=60,
        bm25=Bm25Settings(weights=(4, 4, 4, 2, 2, 2, 1)),
    ),
    "image_bge": Definition(
        id="image_bge",
        label="Image and BGE",
        retrievers=("siglip_image", "bge"),
        candidate_depth=1000,
        rrf_k=60,
    ),
    "image_bm25_bge": Definition(
        id="image_bm25_bge",
        label="Image, BM25, and BGE",
        retrievers=("siglip_image", "bm25", "bge"),
        candidate_depth=1000,
        rrf_k=60,
        bm25=Bm25Settings(weights=(4, 4, 4, 2, 2, 2, 1)),
    ),
}


def id_of(name: str) -> RecipeId:
    try:
        return _ALIASES[name]
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
