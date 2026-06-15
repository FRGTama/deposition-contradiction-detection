from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from .llm_client import llm_chat
from .logger import debug_log
from .models import ContradictionType, ExtractedClaim

Compatibility = Literal["compatible", "incompatible", "uncertain"]


@dataclass(frozen=True)
class ClaimPairClassification:
    type: ContradictionType
    compatibility: Compatibility
    confidence: float
    rationale: str


async def classify_claim_pair_with_llm(
    claim1: ExtractedClaim,
    claim2: ExtractedClaim,
    semantic_score: float,
) -> ClaimPairClassification:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    payload = _classifier_payload(claim1, claim2, semantic_score)

    debug_log("llm_classifier.input", {"provider": provider, **payload})

    prompt = _build_classifier_prompt(payload)
    model, raw_text = await llm_chat(prompt, max_tokens=900, label="classifier")

    result = _parse_classification(raw_text)
    _log_output(provider, model, result, raw_text)
    return result


def _classifier_payload(
    claim1: ExtractedClaim,
    claim2: ExtractedClaim,
    semantic_score: float,
) -> dict[str, Any]:
    return {
        "semantic_similarity": round(semantic_score, 4),
        "claim_a": claim1.model_dump(by_alias=True, exclude_none=True),
        "claim_b": claim2.model_dump(by_alias=True, exclude_none=True),
    }


def _build_classifier_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a contradiction detection system for legal deposition claims.

Your task is to compare two extracted factual claims and decide whether they are contradictory.

Contradiction labels:
1. direct_contradiction: the two claims explicitly cannot both be true.
2. inferential_contradiction: the claims do not directly oppose each other, but their facts cannot reasonably coexist.
3. false_positive: the claims look related but can both be true.
4. neutral_or_insufficient: there is not enough information to decide.

Rules:
1. Compare subject, relation, object, time, location, negation, and certainty.
2. Do not mark contradiction if the time periods are different.
3. Do not mark contradiction if both claims can reasonably be true.
4. Treat uncertainty carefully.
5. A memory failure is not automatically a contradiction.
6. Explain the decision using only the provided claims.
7. Return valid JSON only.
8. When one claim denies knowledge of, or familiarity with, a location (e.g. "never heard of X", "don't know where X is"), and the other claim asserts presence, movement, or familiarity in a related location, this is an inferential contradiction because the denial of any knowledge implies the witness could not have affirmatively been in the vicinity.
9. Claims about the same location but with different levels of specificity (e.g. "warehouse" vs "area" vs "that part of town") should be treated as referring to overlapping locations when they share identifiable place names.

Output schema:
{{
  "pair_id": "...",
  "relatedness": "same_event" | "same_topic" | "weakly_related" | "unrelated",
  "contradiction_label": "direct_contradiction" | "inferential_contradiction" | "false_positive" | "neutral_or_insufficient",
  "contradiction_score": 0.0,
  "reason": "...",
  "conflicting_fields": ["time", "location", "object", "negation"],
  "claim_a_summary": "...",
  "claim_b_summary": "...",
  "needs_human_review": true
}}

Input:
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def _parse_classification(text: str) -> ClaimPairClassification:
    json_text = _extract_json_object_text(text)
    if not json_text:
        return ClaimPairClassification(
            type="FALSE_POSITIVE",
            compatibility="uncertain",
            confidence=0.0,
            rationale="Classifier did not return valid JSON.",
        )

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return ClaimPairClassification(
            type="FALSE_POSITIVE",
            compatibility="uncertain",
            confidence=0.0,
            rationale="Classifier JSON could not be parsed.",
        )

    claim_type = _normalize_classifier_type(parsed.get("contradiction_label") or parsed.get("type"))
    if claim_type not in {"DIRECT", "INFERENTIAL", "FALSE_POSITIVE"}:
        claim_type = "FALSE_POSITIVE"

    compatibility = parsed.get("compatibility")
    if compatibility not in {"compatible", "incompatible", "uncertain"}:
        compatibility = _compatibility_from_type(claim_type, parsed.get("contradiction_label"))

    confidence = parsed.get("contradiction_score", parsed.get("confidence", 0))
    try:
        confidence_float = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_float = 0.0

    rationale = parsed.get("reason") or parsed.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = "Classifier did not provide a rationale."

    return ClaimPairClassification(
        type=claim_type,
        compatibility=compatibility,
        confidence=confidence_float,
        rationale=" ".join(rationale.split()),
    )


def _normalize_classifier_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "direct": "DIRECT",
        "direct_contradiction": "DIRECT",
        "inferential": "INFERENTIAL",
        "inferential_contradiction": "INFERENTIAL",
        "false_positive": "FALSE_POSITIVE",
        "neutral_or_insufficient": "FALSE_POSITIVE",
        "neutral": "FALSE_POSITIVE",
        "insufficient": "FALSE_POSITIVE",
    }
    if text in mapping:
        return mapping[text]
    if value in {"DIRECT", "INFERENTIAL", "FALSE_POSITIVE"}:
        return str(value)
    return "FALSE_POSITIVE"


def _compatibility_from_type(claim_type: ContradictionType, raw_label: Any) -> Compatibility:
    if claim_type in {"DIRECT", "INFERENTIAL"}:
        return "incompatible"
    label = str(raw_label or "").lower()
    if "insufficient" in label or "neutral" in label:
        return "uncertain"
    return "compatible"


def _extract_json_object_text(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed
    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start < 0 or end < start:
        return ""
    return trimmed[start : end + 1]


def _log_output(
    provider: str,
    model: str,
    result: ClaimPairClassification,
    raw_text: str,
) -> None:
    debug_log(
        "llm_classifier.output",
        {
            "provider": provider,
            "model": model,
            "rawText": raw_text,
            "classification": {
                "type": result.type,
                "compatibility": result.compatibility,
                "confidence": result.confidence,
                "rationale": result.rationale,
            },
        },
    )
