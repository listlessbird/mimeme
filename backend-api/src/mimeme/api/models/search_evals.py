from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field, field_validator

from mimeme import search
from mimeme.search_eval.model import Intent, QuerySource


class CreateSearchEvalQueryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    intent: Intent
    source: QuerySource = "human"


class AddSearchEvalCandidateRequest(BaseModel):
    image_id: int = Field(gt=0)


class PoolSearchEvalQueryRequest(BaseModel):
    recipe_ids: tuple[search.recipe.RecipeId, ...] = Field(min_length=1)


class SaveSearchEvalJudgmentRequest(BaseModel):
    grade: int = Field(ge=0, le=3)
    revision: int = Field(ge=0)


class CreateSearchEvalRunRequest(BaseModel):
    recipe_id: search.recipe.RecipeId = Field(validation_alias=AliasChoices("recipe_id", "mode"))

    @field_validator("recipe_id", mode="before")
    @classmethod
    def _resolve_recipe(cls, value: object) -> search.recipe.RecipeId:
        if not isinstance(value, str):
            raise ValueError("recipe ID must be a string")
        return search.recipe.id_of(value)


class CreateSearchEvalExperimentRequest(BaseModel):
    recipe_ids: tuple[search.recipe.RecipeId, ...] = Field(min_length=1)
