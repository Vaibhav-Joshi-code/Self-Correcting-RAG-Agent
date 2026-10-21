"""Strict evaluator and public API contracts."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Score = Annotated[int, Field(strict=True, ge=1, le=5)]
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    retrieval_relevance: Score
    grounding: Score
    hallucination_risk: Score
    completeness: Score
    retry_required: bool
    reason: str = Field(min_length=1, max_length=3000)
    improved_query: str = Field(max_length=4000)
    missing_information: list[str] = Field(max_length=20)
    unsupported_claims: list[str] = Field(max_length=20)


class Source(BaseModel):
    id: str
    source: str
    chunk_id: str
    content: str
    distance: float


class Attempt(BaseModel):
    attempt: int
    search_query: str
    retrieval_k: int
    answer: str
    sources: list[Source]
    evaluation: Evaluation
    accepted: bool
    validation_errors: list[str]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Question


class AskResponse(BaseModel):
    question: str
    answer: str
    status: Literal["accepted", "abstained"]
    accepted: bool
    attempts: int
    sources: list[Source]
    evaluation: Evaluation
    history: list[Attempt]
    stop_reason: Literal["validation_passed", "retry_budget_exhausted"]
    request_id: str
    trace_id: str | None = None
    elapsed_seconds: float
