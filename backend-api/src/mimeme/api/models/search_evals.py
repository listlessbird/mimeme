from __future__ import annotations

from pydantic import BaseModel, Field

from mimeme.search_eval.model import Intent, QuerySource, RunMode


class CreateSearchEvalQueryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    intent: Intent
    source: QuerySource = "human"


class AddSearchEvalCandidateRequest(BaseModel):
    image_id: int = Field(gt=0)


class SaveSearchEvalJudgmentRequest(BaseModel):
    grade: int = Field(ge=0, le=3)
    revision: int = Field(ge=0)


class CreateSearchEvalRunRequest(BaseModel):
    mode: RunMode
