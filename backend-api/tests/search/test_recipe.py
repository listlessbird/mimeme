from __future__ import annotations

import pytest

from mimeme.search import recipe


def test_legacy_names_resolve_to_versioned_definitions() -> None:
    assert recipe.resolve("image") == recipe.resolve("image_only")
    assert recipe.resolve("hybrid") == recipe.resolve("image_siglip_text")
    assert recipe.resolve("hybrid").model_dump() == {
        "id": "image_siglip_text",
        "version": 1,
        "label": "Image and SigLIP text",
        "retrievers": ("siglip_image", "siglip_text"),
        "candidate_depth": 1000,
        "rrf_k": 60,
        "bm25": None,
    }


def test_unknown_and_unavailable_recipes_raise_typed_errors() -> None:
    with pytest.raises(recipe.UnknownRecipe, match="unknown search recipe"):
        recipe.resolve("invented")
    with pytest.raises(recipe.UnavailableRetriever, match="siglip_text"):
        recipe.resolve("hybrid", available={"siglip_image"})


def test_recipe_list_is_closed_and_stable() -> None:
    assert [definition.id for definition in recipe.all()] == [
        "image_only",
        "image_siglip_text",
        "image_bm25",
        "image_bge",
        "image_bm25_bge",
    ]
    assert recipe.resolve("image_bm25").retrievers == ("siglip_image", "bm25")
    assert recipe.resolve("image_bm25").bm25 == recipe.Bm25Settings(weights=(4, 4, 4, 2, 2, 2, 1))
