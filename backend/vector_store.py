from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any

from .embeddings import ClaimEmbedding, claim_to_embedding_text
from .logger import debug_log, text_preview, vector_preview
from .models import ExtractedClaim
from .relation_frames import families_are_comparable

MIN_CANDIDATE_SIMILARITY = 0.55
VERY_LOW_CANDIDATE_SIMILARITY = 0.18


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
        self._collection: Any | None = _create_chroma_collection()

    def add_claims(self, indexed_claims: list[IndexedClaim]) -> None:
        for indexed_claim in indexed_claims:
            self._claims[indexed_claim.claim.id] = indexed_claim

        if self._collection is not None and indexed_claims:
            self._collection.add(
                ids=[indexed_claim.claim.id for indexed_claim in indexed_claims],
                embeddings=[indexed_claim.vector for indexed_claim in indexed_claims],
                documents=[indexed_claim.text for indexed_claim in indexed_claims],
                metadatas=[_chroma_safe_metadata(indexed_claim.metadata) for indexed_claim in indexed_claims],
            )

    def query(
        self,
        claim: ExtractedClaim,
        source: str,
        limit: int,
    ) -> list[tuple[ExtractedClaim, float]]:
        query_claim = self._claims.get(claim.id)
        if not query_claim:
            return []

        if self._collection is not None:
            try:
                result = self._collection.query(
                    query_embeddings=[query_claim.vector],
                    n_results=max(limit * 3, limit),
                    where={"source": source},
                    include=["distances", "metadatas"],
                )
                ids = (result.get("ids") or [[]])[0]
                distances = (result.get("distances") or [[]])[0]
                matches: list[tuple[ExtractedClaim, float]] = []
                for index, claim_id in enumerate(ids):
                    indexed = self._claims.get(str(claim_id))
                    if not indexed or indexed.claim.source != source:
                        continue
                    distance = float(distances[index]) if index < len(distances) else 1.0
                    matches.append((indexed.claim, _score_from_chroma_distance(distance)))
                return matches[:limit]
            except Exception as error:  # pragma: no cover - chroma fallback path
                debug_log("vector.query", {"path": "chroma_error_fallback", "error": str(error)})

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
            "path": "chroma" if vector_store._collection is not None else "memory",
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
                    {"claimId": claim.id, "semanticScore": _round(score)}
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
                    "semanticScore": _round(semantic_score),
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
    entity_keys = _entity_keys(claim)
    location_key = _location_key(claim.location or claim.object)
    time_bucket = _time_bucket(claim)
    object_key = _object_key(claim)
    relation_key = _relation_key(claim.relation)
    topic_key = _topic_key(claim, entity_keys, time_bucket, location_key)
    value_key = _value_key(claim, object_key, location_key, time_bucket)
    return ClaimFrame(
        subject_key=_subject_key(claim.subject),
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
    text = _normalize(subject)
    if not text or text in {"i", "me", "myself", "marcus webb", "witness"}:
        return "witness"
    return _slug(text)


def _relation_key(relation: str) -> str:
    text = _normalize(relation)
    if text in {"heard of", "knew of", "had heard of", "had_heard_of", "know of"}:
        return "had_heard_of"
    if text in {"met face to face", "meet face to face", "met_face_to_face"}:
        return "met_face_to_face"
    return _slug(text)


def _object_key(claim: ExtractedClaim) -> str:
    text = _normalize(" ".join([claim.object, claim.standalone_claim, claim.evidence]))
    if not text:
        return ""
    if "daniel cho" in text:
        return "person:daniel_cho"
    if "victor lane" in text:
        return "person:victor_lane"
    if "hargrove" in text and "warehouse" in text:
        return "place:hargrove_street_warehouse"
    if "hargrove" in text:
        return "place:hargrove_street_area"
    if "honda" in text or "civic" in text:
        return "vehicle:grey_honda_civic"
    if "restaurant" in text:
        return "place:restaurant"
    if "pharmacy" in text:
        return "place:pharmacy"
    if "home" in text or "apartment" in text:
        return "place:home"
    if "pizza" in text or "thai food" in text:
        return "food:ordered_food"
    if "groceries" in text:
        return "errand:groceries"
    if "phone call" in text or "call" in text:
        return "communication:phone_call"
    return _slug(claim.object) or _slug(claim.standalone_claim)[:80]


def _entity_keys(claim: ExtractedClaim) -> list[str]:
    text = _normalize(" ".join([claim.topic, claim.object, claim.location or "", claim.standalone_claim, claim.evidence]))
    entities: list[str] = []

    for marker, entity in [
        ("daniel cho", "person:daniel_cho"),
        ("victor lane", "person:victor_lane"),
        ("honda", "vehicle:grey_honda_civic"),
        ("civic", "vehicle:grey_honda_civic"),
        ("parking lot", "place:parking_lot"),
        ("restaurant", "place:restaurant"),
        ("pharmacy", "place:pharmacy"),
        ("phone call", "communication:phone_call"),
    ]:
        if marker in text and entity not in entities:
            entities.append(entity)

    if "hargrove" in text:
        entities.append("place:hargrove_street_warehouse" if "warehouse" in text else "place:hargrove_street_area")
    if "home" in text or "apartment" in text:
        entities.append("place:home")

    object_key = _object_key(claim)
    if object_key and object_key not in entities and object_key.split(":", 1)[0] in {"person", "place", "vehicle"}:
        entities.append(object_key)

    return entities


def _location_key(value: str) -> str:
    text = _normalize(value)
    if not text:
        return ""
    if "home" in text or "apartment" in text:
        return "place:home"
    if "restaurant" in text:
        return "place:restaurant"
    if "pharmacy" in text:
        return "place:pharmacy"
    if "parking lot" in text:
        return "place:parking_lot"
    if "hargrove" in text and "warehouse" in text:
        return "place:hargrove_street_warehouse"
    if "hargrove" in text:
        return "place:hargrove_street_area"
    return _slug(text)


def _time_bucket(claim: ExtractedClaim) -> str:
    time = claim.time
    text = _normalize(" ".join([claim.topic, claim.standalone_claim, claim.evidence, time.raw if time else ""]))
    if "november 3" in text or "nov 3" in text:
        day = "nov_3"
    elif "monday" in text:
        day = "monday"
    elif "tuesday" in text:
        day = "tuesday"
    else:
        day = ""

    if "all evening" in text or "whole night" in text or "evening" in text or "night" in text:
        period = "evening"
    elif time and time.minutes is not None:
        period = f"minute_{time.minutes}"
    else:
        period = ""

    return "_".join(part for part in [day, period] if part)


def _topic_key(claim: ExtractedClaim, entity_keys: list[str], time_bucket: str, location_key: str) -> str:
    if claim.relation_family == "knowledge" and entity_keys:
        return f"{entity_keys[0]}_prior_knowledge"
    if claim.relation_family == "contact":
        return "contact_with_others"
    if claim.relation_family == "sleep_time":
        return "sleep_time"
    if claim.relation_family in {"location_at_time", "movement"} and (time_bucket or location_key):
        return "location_movement_context"
    if claim.relation_family == "ownership":
        return "vehicle_ownership"
    return _slug(claim.topic) or claim.relation_family


def _value_key(claim: ExtractedClaim, object_key: str, location_key: str, time_bucket: str) -> str:
    if claim.relation_family == "location_at_time":
        return location_key or object_key
    if claim.relation_family == "sleep_time":
        return time_bucket or object_key
    return object_key or location_key or time_bucket


def _normalize(value: str) -> str:
    return (
        value.lower()
        .replace("_", " ")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
        .replace("didn't", "did not")
        .replace("don't", "do not")
        .strip()
    )


def _slug(value: str) -> str:
    token = ""
    tokens: list[str] = []
    for character in _normalize(value):
        if character.isalnum():
            token += character
        elif token:
            tokens.append(token)
            token = ""
    if token:
        tokens.append(token)
    return "_".join(tokens)


def _create_chroma_collection() -> Any | None:
    try:
        import chromadb  # type: ignore[import-not-found]
        from chromadb.config import Settings  # type: ignore[import-not-found]

        client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
        return client.create_collection(name=f"claims_{uuid.uuid4().hex}")
    except Exception as error:
        debug_log("vector.index", {"path": "memory", "reason": "chromadb_unavailable", "error": str(error)})
        return None


def _chroma_safe_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {key: value if isinstance(value, (str, int, float, bool)) else str(value) for key, value in metadata.items()}


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(min(len(left), len(right))))


def _normalize_cosine(value: float) -> float:
    return max(0, min(1, value))


def _score_from_chroma_distance(distance: float) -> float:
    return _normalize_cosine(1 - distance / 2)


def _round(value: float) -> float:
    return round(value * 100) / 100
