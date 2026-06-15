from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ClaimSource = Literal["first", "second"]
ClaimPolarity = Literal["affirmed", "negated", "unknown"]
ClaimCertainty = Literal["certain", "uncertain", "denied", "unknown", "does_not_remember"]
RelationFamily = Literal[
    "location_at_time",
    "movement",
    "sleep_time",
    "ownership",
    "knowledge",
    "contact",
    "action",
    "unknown",
]
ContradictionType = Literal["DIRECT", "INFERENTIAL", "FALSE_POSITIVE"]
Severity = Literal["HIGH", "MEDIUM", "LOW"]


class ClaimTime(BaseModel):
    raw: str
    minutes: int | None = None
    approximate: bool


class QuestionContext(BaseModel):
    asked_about: str | None = Field(default=None, alias="askedAbout")
    time: str | None = None
    location: str | None = None

    model_config = {"populate_by_name": True}


class EvidenceDetail(BaseModel):
    question: str
    answer: str


class ExtractedClaim(BaseModel):
    id: str
    source: ClaimSource
    standalone_claim: str = Field(default="", alias="standaloneClaim")
    speaker: str = "witness"
    topic: str
    subject: str
    relation: str
    relation_family: RelationFamily = Field(alias="relationFamily")
    object: str
    polarity: ClaimPolarity
    negation: bool = False
    certainty: ClaimCertainty = "certain"
    question_context: QuestionContext = Field(default_factory=QuestionContext, alias="questionContext")
    time: ClaimTime | None = None
    location: str | None = None
    uncertainty_markers: list[str] = Field(default_factory=list, alias="uncertaintyMarkers")
    evidence: str
    evidence_detail: EvidenceDetail | None = Field(default=None, alias="evidenceDetail")

    model_config = {"populate_by_name": True}


class ClaimPairScore(BaseModel):
    fr: float
    fu: float
    confidence: float
    topic_score: float = Field(alias="topicScore")
    semantic_similarity: float | None = Field(default=None, alias="semanticSimilarity")
    nli_contradiction_score: float | None = Field(default=None, alias="nliContradictionScore")
    structured_mismatch_score: float | None = Field(default=None, alias="structuredMismatchScore")
    final_contradiction_score: float | None = Field(default=None, alias="finalContradictionScore")
    type: ContradictionType
    severity: Severity
    rationale: str
    claim1: ExtractedClaim
    claim2: ExtractedClaim

    model_config = {"populate_by_name": True}


class AnalysisRequest(BaseModel):
    transcript1: str = ""
    transcript2: str = ""


class AnalysisResult(BaseModel):
    claims: list[ExtractedClaim]
    contradictions: list[ClaimPairScore]
    provider: str
    model: str
    embedding_provider: str = Field(alias="embeddingProvider")
    embedding_model: str = Field(alias="embeddingModel")

    model_config = {"populate_by_name": True}
