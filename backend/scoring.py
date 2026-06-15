from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from .embeddings import ClaimEmbedding
from .text_utils import clamp, normalize_text, round_score
from .logger import debug_log
from .llm_classifier import ClaimPairClassification, classify_claim_pair_with_llm
from .models import ClaimPairScore, ContradictionType, ExtractedClaim, Severity
from .vector_store import CandidatePair, ClaimFrame, InMemoryClaimVectorStore, frame_for_claim, index_claims, retrieve_candidate_pairs

HEDGE_MARKERS = {
    "about",
    "around",
    "approximately",
    "maybe",
    "might",
    "think",
    "probably",
    "possibly",
    "i guess",
    "i don't remember",
    "do not remember",
    "does_not_remember",
    "unknown",
    "uncertain",
    "not sure",
}

DIRECT_NEGATIONS = {
    "no",
    "not",
    "never",
    "none",
    "didn't",
    "dont",
    "don't",
}

TOP_K_CANDIDATES = 5
LOW_CONFIDENCE_THRESHOLD = 0.60
LOW_CONFIDENCE_FALSE_POSITIVE_DISPLAY_THRESHOLD = 0.20

_nli_model: Any = None
NLI_LABELS = ["contradiction", "entailment", "neutral"]


def _get_nli_model() -> Any:
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder

        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    return _nli_model


def _batch_nli_scores(pairs: list[tuple[str, str]]) -> list[float]:
    if not pairs:
        return []
    model = _get_nli_model()
    scores = model.predict(pairs, apply_softmax=True)
    return [float(s[0]) for s in scores]


@dataclass(frozen=True)
class FastGuardrailResult:
    type: ContradictionType
    confidence: float
    fr: float
    rationale: str
    guardrail: str


async def score_contradictions(
    claims: list[ExtractedClaim],
    embeddings: list[ClaimEmbedding],
) -> list[ClaimPairScore]:
    case_id = "analysis-request"
    vector_store = InMemoryClaimVectorStore()
    await index_claims(case_id, claims, embeddings, vector_store)
    candidate_pairs = await retrieve_candidate_pairs(
        [claim for claim in claims if claim.source == "first"],
        [claim for claim in claims if claim.source == "second"],
        vector_store,
        top_k=TOP_K_CANDIDATES,
    )

    nli_inputs: list[tuple[str, str]] = []
    for pair in candidate_pairs:
        if _apply_fast_guardrails(pair) is None:
            nli_inputs.append((
                pair.claim1.standalone_claim,
                pair.claim2.standalone_claim,
            ))

    nli_scores = _batch_nli_scores(nli_inputs)

    results: list[ClaimPairScore] = []
    nli_idx = 0

    for pair in candidate_pairs:
        fast_guardrail = _apply_fast_guardrails(pair)
        if fast_guardrail is not None:
            result = _build_result_from_guardrail(pair, fast_guardrail)
            _log_guardrail(pair, fast_guardrail.guardrail, result)
            _log_final(pair, None, result)
            if _should_emit(result):
                results.append(result)
            continue

        nli_contradiction = nli_scores[nli_idx]
        nli_idx += 1

        llm_result = await classify_claim_pair_with_llm(
            pair.claim1,
            pair.claim2,
            pair.semantic_score,
        )
        final_result = _apply_post_llm_guardrails(pair, llm_result, nli_contradiction)
        _log_final(pair, llm_result, final_result)

        if _should_emit(final_result):
            results.append(final_result)

    return _rank_and_deduplicate(results)[:10]


def _apply_fast_guardrails(
    pair: CandidatePair,
) -> FastGuardrailResult | None:
    claim1 = pair.claim1
    claim2 = pair.claim2

    if claim1.source == claim2.source:
        return _non_emit_guardrail("same_source", "Claims from the same deposition source are not compared.")

    if not claim1.evidence.strip() or not claim2.evidence.strip():
        return _non_emit_guardrail("missing_evidence", "One or both claims lack exact evidence quotes.")

    frame1 = frame_for_claim(claim1)
    frame2 = frame_for_claim(claim2)

    same_fact = (
        frame1.subject_key == frame2.subject_key
        and frame1.relation_key == frame2.relation_key
        and bool(frame1.value_key)
        and frame1.value_key == frame2.value_key
    )

    if same_fact and _has_conflicting_polarity(claim1, claim2):
        return FastGuardrailResult(
            type="DIRECT",
            confidence=0.94,
            fr=1.0,
            guardrail="same_fact_opposite_polarity",
            rationale=(
                "Guardrail: the claims describe the same normalized subject, relation, and value, "
                "but their polarities conflict."
            ),
        )

    if same_fact and _has_same_known_polarity(claim1, claim2):
        return FastGuardrailResult(
            type="FALSE_POSITIVE",
            confidence=0.24,
            fr=0.0,
            guardrail="same_fact_same_polarity",
            rationale=(
                "Guardrail: the claims describe the same normalized fact with the same polarity, "
                "so they are compatible."
            ),
        )

    location_knowledge = _check_location_knowledge_contradiction(claim1, claim2, frame1, frame2)
    if location_knowledge:
        return location_knowledge

    return None


def _check_location_knowledge_contradiction(
    claim1: ExtractedClaim,
    claim2: ExtractedClaim,
    frame1: ClaimFrame,
    frame2: ClaimFrame,
) -> FastGuardrailResult | None:
    candidates = [
        (claim1, claim2, frame1, frame2),
        (claim2, claim1, frame2, frame1),
    ]

    for neg_claim, aff_claim, neg_frame, aff_frame in candidates:
        if not (
            neg_claim.relation_family == "knowledge"
            and neg_claim.negation
            and aff_claim.relation_family in {"movement", "location_at_time"}
            and not aff_claim.negation
        ):
            continue

        if not neg_frame.location_key or not aff_frame.location_key:
            continue

        if _locations_related(neg_frame.location_key, aff_frame.location_key):
            return FastGuardrailResult(
                type="INFERENTIAL",
                confidence=0.82,
                fr=0.72,
                guardrail="knowledge_vs_presence",
                rationale=(
                    "Guardrail: one claim denies knowledge of a location while the other "
                    "asserts presence or movement in a related location. These cannot "
                    "reasonably coexist."
                ),
            )

    return None


def _locations_related(loc_a: str, loc_b: str) -> bool:
    if loc_a == loc_b:
        return True
    if loc_a in loc_b or loc_b in loc_a:
        return True
    tokens_a = set(loc_a.split("_"))
    tokens_b = set(loc_b.split("_"))
    common = tokens_a & tokens_b - {"place", "person", "vehicle", "area", "warehouse"}
    return len(common) >= 1


def _non_emit_guardrail(guardrail: str, rationale: str) -> FastGuardrailResult:
    return FastGuardrailResult(
        type="FALSE_POSITIVE",
        confidence=0.0,
        fr=0.0,
        guardrail=guardrail,
        rationale=rationale,
    )


def _apply_post_llm_guardrails(
    pair: CandidatePair,
    llm_result: ClaimPairClassification,
    nli_contradiction: float = 0.0,
) -> ClaimPairScore:
    claim1 = pair.claim1
    claim2 = pair.claim2
    frame1 = frame_for_claim(claim1)
    frame2 = frame_for_claim(claim2)
    final_type = llm_result.type
    rationale = llm_result.rationale
    guardrail_name = "none"

    same_fact = bool(
        frame1.subject_key == frame2.subject_key
        and frame1.relation_key == frame2.relation_key
        and frame1.value_key
        and frame1.value_key == frame2.value_key
    )

    if same_fact and _has_conflicting_polarity(claim1, claim2):
        final_type = "DIRECT"
        guardrail_name = "post_same_fact_opposite_polarity"
        rationale = (
            "Post-LLM guardrail: same normalized fact with opposite polarity. "
            f"Classifier rationale: {llm_result.rationale}"
        )
    elif (
        llm_result.type in {"DIRECT", "INFERENTIAL"}
        and llm_result.compatibility in {"compatible", "uncertain"}
    ):
        final_type = "FALSE_POSITIVE"
        guardrail_name = "post_low_confidence_or_compatible"
        rationale = (
            "Post-LLM guardrail: classifier found a contradiction but assessed "
            "compatibility as compatible or uncertain. "
            f"Classifier rationale: {llm_result.rationale}"
        )

    if guardrail_name != "none":
        debug_log(
            "guardrail.applied",
            {
                "claim1Id": claim1.id,
                "claim2Id": claim2.id,
                "guardrail": guardrail_name,
                "llmType": llm_result.type,
                "finalType": final_type,
                "llmConfidence": llm_result.confidence,
            },
        )

    components = _score_components(pair, final_type, llm_result.compatibility, nli_contradiction)
    confidence = components["final"]
    fr = _fr_for_type(final_type, confidence)
    fu = _uncertainty_difference(claim1, claim2)

    return ClaimPairScore(
        fr=round_score(fr),
        fu=round_score(fu),
        confidence=round_score(confidence),
        topicScore=round_score(pair.semantic_score),
        semanticSimilarity=round_score(components["semantic"]),
        nliContradictionScore=round_score(components["nli"]),
        structuredMismatchScore=round_score(components["structured"]),
        finalContradictionScore=round_score(components["final"]),
        type=final_type,
        severity=_classify_severity(final_type, confidence),
        rationale=rationale,
        claim1=claim1,
        claim2=claim2,
    )


def _build_result_from_guardrail(
    pair: CandidatePair,
    guardrail: FastGuardrailResult,
) -> ClaimPairScore:
    fu = _uncertainty_difference(pair.claim1, pair.claim2)
    compatibility: Literal["compatible", "incompatible", "uncertain"] = (
        "incompatible" if guardrail.type != "FALSE_POSITIVE" else "compatible"
    )
    components = _score_components(pair, guardrail.type, compatibility)
    confidence = components["final"]
    fr = _fr_for_type(guardrail.type, confidence)

    return ClaimPairScore(
        fr=round_score(fr),
        fu=round_score(fu),
        confidence=round_score(confidence),
        topicScore=round_score(pair.semantic_score),
        semanticSimilarity=round_score(components["semantic"]),
        nliContradictionScore=round_score(components["nli"]),
        structuredMismatchScore=round_score(components["structured"]),
        finalContradictionScore=round_score(components["final"]),
        type=guardrail.type,
        severity=_classify_severity(guardrail.type, confidence),
        rationale=guardrail.rationale,
        claim1=pair.claim1,
        claim2=pair.claim2,
    )


def _score_components(
    pair: CandidatePair,
    contradiction_type: ContradictionType,
    compatibility: Literal["compatible", "incompatible", "uncertain"],
    nli_score: float | None = None,
) -> dict[str, float]:
    semantic = clamp(pair.semantic_score)
    nli = nli_score if nli_score is not None else _nli_contradiction_score(contradiction_type, compatibility)
    structured = _structured_mismatch_score(pair)
    base = clamp(0.25 * semantic + 0.50 * nli + 0.25 * structured)

    uncertainty_penalty = _average_uncertainty_penalty(pair.claim1, pair.claim2)
    final = clamp(base * (1 - 0.3 * uncertainty_penalty))

    return {
        "semantic": semantic,
        "nli": nli,
        "structured": structured,
        "final": final,
    }


def _nli_contradiction_score(
    contradiction_type: ContradictionType,
    compatibility: Literal["compatible", "incompatible", "uncertain"],
) -> float:
    if contradiction_type == "DIRECT":
        return 0.95 if compatibility == "incompatible" else 0.72
    if contradiction_type == "INFERENTIAL":
        return 0.84 if compatibility == "incompatible" else 0.62
    if compatibility == "incompatible":
        return 0.18
    return 0.0


def _structured_mismatch_score(pair: CandidatePair) -> float:
    claim1 = pair.claim1
    claim2 = pair.claim2
    frame1 = frame_for_claim(claim1)
    frame2 = frame_for_claim(claim2)

    same_subject = frame1.subject_key and frame1.subject_key == frame2.subject_key
    same_relation = frame1.relation_key and frame1.relation_key == frame2.relation_key
    same_value = frame1.value_key and frame1.value_key == frame2.value_key
    same_time = frame1.time_bucket and frame1.time_bucket == frame2.time_bucket
    same_topic = frame1.topic_key and frame1.topic_key == frame2.topic_key

    if same_subject and same_relation and same_value and _has_conflicting_polarity(claim1, claim2):
        return 1.0

    if same_subject and same_relation and same_value and _has_same_known_polarity(claim1, claim2):
        return 0.0

    if (
        same_subject
        and same_time
        and claim1.relation_family == "location_at_time"
        and claim2.relation_family == "location_at_time"
        and frame1.location_key
        and frame2.location_key
        and frame1.location_key != frame2.location_key
    ):
        return 0.9

    if same_subject and same_topic and {claim1.relation_family, claim2.relation_family} == {"location_at_time", "movement"}:
        return 0.78

    text = _claim_text(claim1) + " " + _claim_text(claim2)
    if "sleep" in text and any(term in text for term in ["phone call", "called", "made a call"]):
        return 0.8

    if {claim1.relation_family, claim2.relation_family} == {"knowledge", "contact"}:
        return 0.0

    if same_subject and same_topic and frame1.object_key and frame2.object_key and frame1.object_key != frame2.object_key:
        return 0.45

    return 0.15 if same_subject or same_topic else 0.0


def _fr_for_type(contradiction_type: ContradictionType, confidence: float) -> float:
    if contradiction_type == "DIRECT":
        return max(0.78, confidence)
    if contradiction_type == "INFERENTIAL":
        return max(0.58, confidence * 0.86)
    return 0.0


def _should_emit(result: ClaimPairScore) -> bool:
    if result.confidence <= 0:
        return False
    if result.type == "FALSE_POSITIVE":
        return os.getenv("DEBUG_PIPELINE") == "true" or result.confidence >= LOW_CONFIDENCE_FALSE_POSITIVE_DISPLAY_THRESHOLD
    return True


def _rank_and_deduplicate(results: list[ClaimPairScore]) -> list[ClaimPairScore]:
    type_rank = {"DIRECT": 0, "INFERENTIAL": 1, "FALSE_POSITIVE": 2}
    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    seen: set[tuple[str, str]] = set()
    unique: list[ClaimPairScore] = []

    for result in sorted(
        results,
        key=lambda item: (
            type_rank[item.type],
            -item.confidence,
            -item.topic_score,
            severity_rank[item.severity],
            item.fu,
        ),
    ):
        key = (result.claim1.id, result.claim2.id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)

    return unique


def _classify_severity(
    contradiction_type: ContradictionType,
    confidence: float,
) -> Severity:
    if contradiction_type == "DIRECT" and confidence >= 0.72:
        return "HIGH"
    if contradiction_type != "FALSE_POSITIVE" and confidence >= 0.48:
        return "MEDIUM"
    return "LOW"


def _certainty_from_claim(claim: ExtractedClaim) -> float:
    text = _claim_text(claim)
    markers = [normalize_text(marker) for marker in claim.uncertainty_markers] + _tokenize(text)
    hedge_count = sum(1 for marker in markers if marker in HEDGE_MARKERS)
    direct_count = sum(1 for marker in markers if marker in DIRECT_NEGATIONS)
    approximate_time = bool(claim.time and claim.time.approximate)
    certainty = 0.78 + direct_count * 0.08 - hedge_count * 0.16 - (0.08 if approximate_time else 0)
    return clamp(certainty, 0.22, 0.98)


def _uncertainty_difference(claim1: ExtractedClaim, claim2: ExtractedClaim) -> float:
    return abs(_certainty_from_claim(claim1) - _certainty_from_claim(claim2))


def _average_uncertainty_penalty(claim1: ExtractedClaim, claim2: ExtractedClaim) -> float:
    return 1 - (_certainty_from_claim(claim1) + _certainty_from_claim(claim2)) / 2


def _has_conflicting_polarity(claim1: ExtractedClaim, claim2: ExtractedClaim) -> bool:
    return {claim1.polarity, claim2.polarity} == {"affirmed", "negated"}


def _has_same_known_polarity(claim1: ExtractedClaim, claim2: ExtractedClaim) -> bool:
    return claim1.polarity != "unknown" and claim1.polarity == claim2.polarity


def _claim_text(claim: ExtractedClaim) -> str:
    return normalize_text(
        " ".join(
            [
                claim.topic,
                claim.standalone_claim,
                claim.subject,
                claim.relation,
                claim.object,
                claim.location or "",
                claim.evidence,
                claim.certainty,
                " ".join(claim.uncertainty_markers),
            ]
        )
    )


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    token = ""
    for character in normalize_text(text):
        if character.isalnum() or character == "'":
            token += character
        elif token:
            if len(token) > 2:
                tokens.append(token)
            token = ""
    if token and len(token) > 2:
        tokens.append(token)
    return tokens




def _log_guardrail(pair: CandidatePair, guardrail: str, result: ClaimPairScore) -> None:
    debug_log(
        "guardrail.applied",
        {
            "claim1Id": pair.claim1.id,
            "claim2Id": pair.claim2.id,
            "semanticScore": pair.semantic_score,
            "guardrail": guardrail,
            "finalType": result.type,
            "confidence": result.confidence,
            "rationale": result.rationale,
        },
    )


def _log_final(
    pair: CandidatePair,
    llm_result: ClaimPairClassification | None,
    result: ClaimPairScore,
) -> None:
    debug_log(
        "scoring.final",
        {
            "claim1Id": pair.claim1.id,
            "claim2Id": pair.claim2.id,
            "semanticScore": pair.semantic_score,
            "relationFamilies": [pair.claim1.relation_family, pair.claim2.relation_family],
            "llmClassification": llm_result.type if llm_result else None,
            "llmCompatibility": llm_result.compatibility if llm_result else None,
            "llmConfidence": llm_result.confidence if llm_result else None,
            "finalClassification": result.type,
            "confidence": result.confidence,
            "semanticSimilarity": result.semantic_similarity,
            "nliContradictionScore": result.nli_contradiction_score,
            "structuredMismatchScore": result.structured_mismatch_score,
            "finalContradictionScore": result.final_contradiction_score,
            "severity": result.severity,
            "rationale": result.rationale,
        },
    )
    debug_log(
        "scoring.pair",
        {
            "claim1Id": pair.claim1.id,
            "claim2Id": pair.claim2.id,
            "semanticScore": result.topic_score,
            "semanticSimilarity": result.semantic_similarity,
            "nliContradictionScore": result.nli_contradiction_score,
            "structuredMismatchScore": result.structured_mismatch_score,
            "finalContradictionScore": result.final_contradiction_score,
            "fr": result.fr,
            "fu": result.fu,
            "confidence": result.confidence,
            "type": result.type,
            "severity": result.severity,
            "rationale": result.rationale,
        },
    )
