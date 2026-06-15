from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .embeddings import ClaimEmbedding, claim_to_embedding_text
from .logger import debug_log, text_preview, vector_preview
from .models import ExtractedClaim
from .relation_frames import FAMILY_CONSTRAINTS, families_are_comparable
from .text_utils import normalize_text, round_score, slugify

MIN_CANDIDATE_SIMILARITY = 0.55
VERY_LOW_CANDIDATE_SIMILARITY = 0.18
WITNESS_ALIASES = {"i", "me", "my", "myself", "we", "us", "witness"}
ENTITY_STOPWORDS = {
    "a", "an", "and", "at", "before", "did", "for", "from",
    "had", "has", "have", "he", "her", "him", "his", "i", "in",
    "it", "me", "my", "of", "on", "or", "she", "the", "they",
    "to", "was", "we", "were", "witness", "you",
}
PERIOD_WORDS = {"morning", "afternoon", "evening", "night", "midnight", "noon"}
WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


@dataclass(frozen=True)
class ClaimFrame:
    subject_key: str
    relation_key: str
    object_key: str
    entity_keys: tuple[str, ...]
    topic_key: str
    time_bucket: str
    location_key: str
    value_key: str
    relation_family: str


@dataclass(frozen=True)
class CandidatePair:
    claim1: ExtractedClaim
    claim2: ExtractedClaim
    semantic_score: float
    filter_reasons: tuple[str, ...]


@dataclass(frozen=True)
class IndexedClaim:
    claim: ExtractedClaim
    vector: list[float]
    metadata: dict[str, Any]
    text: str


class InMemoryClaimVectorStore:
    def __init__(self) -> None:
        self._claims: dict[str, IndexedClaim] = {}

    def add_claims(self, indexed_claims: list[IndexedClaim]) -> None:
        for indexed_claim in indexed_claims:
            self._claims[indexed_claim.claim.id] = indexed_claim

    def query(
        self,
        claim: ExtractedClaim,
        source: str,
        limit: int,
    ) -> list[tuple[ExtractedClaim, float]]:
        query_claim = self._claims.get(claim.id)
        if not query_claim:
            return []

        candidates = [indexed for indexed in self._claims.values() if indexed.claim.source == source]
        scored = [
            (indexed.claim, _normalize_cosine(_cosine_similarity(query_claim.vector, indexed.vector)))
            for indexed in candidates
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


async def index_claims(
    case_id: str,
    claims: list[ExtractedClaim],
    embeddings: list[ClaimEmbedding],
    vector_store: InMemoryClaimVectorStore,
) -> None:
    vectors = {embedding.claim_id: _normalize_vector(embedding.vector) for embedding in embeddings}
    indexed_claims: list[IndexedClaim] = []

    for claim in claims:
        vector = vectors.get(claim.id)
        if vector is None:
            continue
        text = claim_to_embedding_text(claim)
        metadata = _metadata_for_claim(case_id, claim)
        indexed_claims.append(IndexedClaim(claim=claim, vector=vector, metadata=metadata, text=text))

    vector_store.add_claims(indexed_claims)
    debug_log(
        "vector.index",
        {
            "caseId": case_id,
            "count": len(indexed_claims),
            "path": "memory",
            "claims": [
                {
                    "claimId": indexed.claim.id,
                    "source": indexed.claim.source,
                    "relationFamily": indexed.claim.relation_family,
                    "textPreview": text_preview(indexed.text),
                    "vector": vector_preview(indexed.vector),
                    "metadata": indexed.metadata,
                }
                for indexed in indexed_claims
            ],
        },
    )


async def retrieve_candidate_pairs(
    first_claims: list[ExtractedClaim],
    second_claims: list[ExtractedClaim],
    vector_store: InMemoryClaimVectorStore,
    top_k: int = 5,
) -> list[CandidatePair]:
    second_claim_ids = {claim.id for claim in second_claims}
    seen: set[tuple[str, str]] = set()
    pairs: list[CandidatePair] = []
    filtered_count = 0

    for first_claim in first_claims:
        matches = vector_store.query(first_claim, "second", top_k)
        debug_log(
            "vector.query",
            {
                "claimId": first_claim.id,
                "source": first_claim.source,
                "targetSource": "second",
                "topK": top_k,
                "matches": [
                    {"claimId": claim.id, "semanticScore": round_score(score)}
                    for claim, score in matches
                ],
            },
        )

        for second_claim, semantic_score in matches:
            if second_claim.id not in second_claim_ids:
                continue
            key = (first_claim.id, second_claim.id)
            if key in seen:
                continue
            seen.add(key)

            keep, reasons = _candidate_filter_reasons(first_claim, second_claim, semantic_score)
            event = "candidate.generated" if keep else "candidate.filtered"
            debug_log(
                event,
                {
                    "claim1Id": first_claim.id,
                    "claim2Id": second_claim.id,
                    "semanticScore": round_score(semantic_score),
                    "relationFamilies": [first_claim.relation_family, second_claim.relation_family],
                    "reasons": reasons,
                },
            )
            if keep:
                pairs.append(
                    CandidatePair(
                        claim1=first_claim,
                        claim2=second_claim,
                        semantic_score=semantic_score,
                        filter_reasons=tuple(reasons),
                    )
                )
            else:
                filtered_count += 1

    if filtered_count == 0:
        debug_log(
            "candidate.filtered",
            {
                "count": 0,
                "reason": "no_candidate_pairs_filtered",
            },
        )

    return sorted(pairs, key=lambda pair: pair.semantic_score, reverse=True)


def _candidate_filter_reasons(
    claim1: ExtractedClaim,
    claim2: ExtractedClaim,
    semantic_score: float,
) -> tuple[bool, list[str]]:
    if claim1.source == claim2.source:
        return False, ["same_source"]
    if not claim1.evidence.strip() or not claim2.evidence.strip():
        return False, ["missing_evidence"]

    frame1 = frame_for_claim(claim1)
    frame2 = frame_for_claim(claim2)
    reasons: list[str] = []

    if semantic_score >= MIN_CANDIDATE_SIMILARITY:
        reasons.append("high_semantic_similarity")

    if frame1.subject_key and frame1.subject_key == frame2.subject_key:
        reasons.append("same_subject")
    if set(frame1.entity_keys) & set(frame2.entity_keys):
        reasons.append("shared_entity")
    if frame1.topic_key and frame1.topic_key == frame2.topic_key:
        reasons.append("same_topic")
    if frame1.relation_family == frame2.relation_family:
        reasons.append("same_relation_family")
    elif families_are_comparable(frame1.relation_family, frame2.relation_family):
        reasons.append("compatible_relation_families")
    if frame1.time_bucket and frame1.time_bucket == frame2.time_bucket:
        reasons.append("overlapping_time_context")
    if frame1.location_key and frame1.location_key == frame2.location_key:
        reasons.append("same_location")

    if semantic_score < VERY_LOW_CANDIDATE_SIMILARITY and not reasons:
        return False, ["very_low_similarity"]

    if reasons:
        return True, reasons

    return False, ["no_candidate_signal"]


def _metadata_for_claim(case_id: str, claim: ExtractedClaim) -> dict[str, Any]:
    return {
        "caseId": case_id,
        "claimId": claim.id,
        "source": claim.source,
        "relation": claim.relation,
        "relationFamily": claim.relation_family,
        "subject": claim.subject,
        "object": claim.object,
        "standaloneClaim": claim.standalone_claim,
        "timeRaw": claim.time.raw if claim.time else "",
        "polarity": claim.polarity,
        "evidence": claim.evidence,
    }


def frame_for_claim(claim: ExtractedClaim) -> ClaimFrame:
    subject_key = _subject_key(claim.subject)
    relation_key = _relation_key(claim)
    object_key = _object_key(claim)
    location_key = _location_key(claim.location or "")
    entity_keys = _entity_keys(claim)
    time_bucket = _time_bucket(claim)
    topic_key = _topic_key(claim, entity_keys, object_key, location_key, time_bucket)
    value_key = _value_key(claim, object_key, location_key, time_bucket)
    return ClaimFrame(
        subject_key=subject_key,
        relation_key=relation_key,
        object_key=object_key,
        entity_keys=tuple(entity_keys),
        topic_key=topic_key,
        time_bucket=time_bucket,
        location_key=location_key,
        value_key=value_key,
        relation_family=claim.relation_family,
    )


def _subject_key(subject: str) -> str:
    text = normalize_text(subject)
    if not text or text in WITNESS_ALIASES:
        return "witness"
    return slugify(text)


def _relation_key(claim: ExtractedClaim) -> str:
    return claim.relation_family


def _object_key(claim: ExtractedClaim) -> str:
    return _semantic_key(claim.object) or _semantic_key(claim.location or "") or _fallback_claim_key(claim)


def _entity_keys(claim: ExtractedClaim) -> list[str]:
    entities: list[str] = []

    for value in [claim.object, claim.location or ""]:
        sk = _semantic_key(value)
        if sk:
            entities.append(sk)
        entities.extend(_named_phrase_keys(value))

    for value in [claim.standalone_claim, claim.evidence]:
        entities.extend(_named_phrase_keys(value))

    return _unique(entities)


def _location_key(value: str) -> str:
    return _semantic_key(value)


def _time_bucket(claim: ExtractedClaim) -> str:
    time = claim.time
    values = [
        time.raw if time else "",
        claim.question_context.time or "",
        claim.topic,
        claim.standalone_claim,
        claim.evidence,
    ]
    text = normalize_text(" ".join(values))
    explicit_parts = _time_parts_from_text(text)

    if time and time.minutes is not None:
        explicit_parts.append(f"minute_{time.minutes}")

    return "_".join(_unique(explicit_parts))


def _topic_key(
    claim: ExtractedClaim,
    entity_keys: list[str],
    object_key: str,
    location_key: str,
    time_bucket: str,
) -> str:
    explicit = _explicit_topic_key(claim)
    if explicit:
        return explicit

    anchor = entity_keys[0] if entity_keys else object_key or location_key or time_bucket
    return "_".join(p for p in _unique([claim.relation_family, anchor]) if p)


def _value_key(claim: ExtractedClaim, object_key: str, location_key: str, time_bucket: str) -> str:
    required_slots = set(FAMILY_CONSTRAINTS.get(claim.relation_family, {}).get("required_slots", ()))
    ordered_values = []

    if "location" in required_slots:
        ordered_values.append(location_key)
    if "object" in required_slots:
        ordered_values.append(object_key)
    if "time" in required_slots:
        ordered_values.append(time_bucket)

    ordered_values.extend([object_key, location_key, time_bucket])
    return "_".join(_unique(value for value in ordered_values if value))


def _semantic_key(value: str, max_tokens: int | None = None) -> str:
    tokens = _meaningful_tokens(value)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return "_".join(tokens)


def _explicit_topic_key(claim: ExtractedClaim) -> str:
    topic_text = normalize_text(claim.topic)
    if not topic_text:
        return ""
    if topic_text in {normalize_text(claim.standalone_claim), normalize_text(claim.evidence)}:
        return ""
    if len(_meaningful_tokens(claim.topic)) > 5:
        return ""
    return _semantic_key(claim.topic)


def _fallback_claim_key(claim: ExtractedClaim) -> str:
    text = claim.standalone_claim or claim.evidence or claim.topic
    return "_".join(_meaningful_tokens(text)[:8])


def _named_phrase_keys(value: str) -> list[str]:
    keys: list[str] = []
    for phrase in _capitalized_phrases(value):
        key = _semantic_key(phrase)
        if key:
            keys.append(key)

    for quoted in re.findall(r'"([^"]+)"|' + r"'([^']+)'", value):
        phrase = quoted[0] or quoted[1]
        key = _semantic_key(phrase)
        if key:
            keys.append(key)

    return keys


def _capitalized_phrases(value: str) -> list[str]:
    return [
        match.group(0)
        for match in re.finditer(r"\b(?:[A-Z][a-zA-Z0-9'.-]*)(?:\s+[A-Z][a-zA-Z0-9'.-]*)*\b", value)
        if _is_named_phrase(match.group(0))
    ]


def _is_named_phrase(value: str) -> bool:
    text = normalize_text(value)
    if text in ENTITY_STOPWORDS or text in MONTHS or text in WEEKDAYS or text in PERIOD_WORDS:
        return False
    if text in {"am", "pm", "a m", "p m"}:
        return False
    return bool(_meaningful_tokens(text))


def _time_parts_from_text(text: str) -> list[str]:
    parts: list[str] = []

    for weekday in WEEKDAYS:
        if re.search(rf"\b{weekday}\b", text):
            parts.append(weekday)

    for month in MONTHS:
        match = re.search(rf"\b{month}\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(?P<year>\d{{4}}))?\b", text)
        if match:
            parts.append("_".join(part for part in [month, match.group("day"), match.group("year") or ""] if part))

    for match in re.finditer(r"\b(?P<month>\d{1,2})[/-](?P<day>\d{1,2})(?:[/-](?P<year>\d{2,4}))?\b", text):
        parts.append("_".join(part for part in [match.group("year") or "", match.group("month"), match.group("day")] if part))

    for period in PERIOD_WORDS:
        if re.search(rf"\b{period}\b", text):
            parts.append(period)

    if re.search(r"\b(all|whole|entire)\s+(night|evening|morning|afternoon|day)\b", text):
        parts.append("extended_period")

    return parts


def _meaningful_tokens(value: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-zA-Z0-9]+", normalize_text(value)):
        if len(token) <= 1 or token in ENTITY_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(min(len(left), len(right))))


def _normalize_cosine(value: float) -> float:
    return max(0, min(1, value))
