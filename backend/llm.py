from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from .llm_client import llm_chat
from .logger import debug_log
from .models import ClaimPolarity, ClaimSource, ExtractedClaim
from .qa import QABlock, split_qa_blocks
from .relation_frames import assign_relation_family


async def extract_claims(transcript1: str, transcript2: str) -> dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    debug_log(
        "config.llm",
        {
            "provider": provider,
            "ollamaBaseUrl": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "ollamaModel": os.getenv("OLLAMA_MODEL", "llama3.2"),
            "anthropicModel": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            "openaiModel": os.getenv("OPENAI_MODEL", "gpt-4.1"),
            "deepseekBaseUrl": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "deepseekModel": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        },
    )

    first_blocks = split_qa_blocks(transcript1, "first")
    second_blocks = split_qa_blocks(transcript2, "second")

    first_task = _extract_claims_for_source(first_blocks, "first", provider)
    second_task = _extract_claims_for_source(second_blocks, "second", provider)
    first, second = await asyncio.gather(first_task, second_task)

    return {
        "provider": provider,
        "model": first["model"] if first["model"] == second["model"] else f'{first["model"]}, {second["model"]}',
        "claims": first["claims"] + second["claims"],
        "blocks": first_blocks + second_blocks,
        "claimBlocks": {**first["claimBlocks"], **second["claimBlocks"]},
    }


async def _extract_claims_for_source(
    blocks: list[QABlock],
    source: ClaimSource,
    provider: str,
) -> dict[str, Any]:
    if not blocks:
        return {"model": "", "claims": [], "claimBlocks": {}}

    prompt = _build_batch_extraction_prompt(blocks)
    model, raw_text = await llm_chat(prompt, max_tokens=16000, label="extraction")

    all_claims: list[ExtractedClaim] = []
    claim_blocks: dict[str, str] = {}

    blocks_data = _parse_batch_json(raw_text)

    for block_data in blocks_data:
        block_id = block_data.get("block_id", "")
        raw_claims = block_data.get("claims", [])
        if not isinstance(raw_claims, list):
            raw_claims = []

        matching_block = next((b for b in blocks if b.id == block_id), None)
        block_claims = _normalize_claims(
            raw_claims,
            forced_source=source,
            evidence_question=matching_block.question if matching_block else "",
            evidence_answer=matching_block.answer if matching_block else "",
            block_id=block_id,
        )
        for claim in block_claims:
            claim_blocks[claim.id] = block_id

        _log_llm_output(provider, model, raw_text, raw_claims, block_claims, source, block_id)
        all_claims.extend(block_claims)

    return {"model": model, "claims": all_claims, "claimBlocks": claim_blocks}


def _build_batch_extraction_prompt(blocks: list[QABlock]) -> str:
    blocks_text = "\n\n".join(
        f"--- Block {block.id} ---\nQ: {block.question}\nA: {block.answer}"
        for block in blocks
    )

    return f"""
You are an information extraction system for legal deposition transcripts.

Your task is to extract atomic factual claims from the Q/A blocks below.

Return valid JSON only. Return exactly one object with a "blocks" array.

Rules:
1. Use both the question and the answer.
2. Do not extract from the answer alone.
3. Convert implicit answers into complete standalone claims.
4. Extract only factual claims made or confirmed by the witness.
5. Do not extract attorney assumptions unless the witness accepts or confirms them.
6. Split compound answers into separate atomic claims.
7. Preserve time, location, negation, uncertainty, and speaker.
8. If the answer is vague, mark missing fields as null or "".
9. If the witness says "I don't know", "I don't remember", or refuses to answer, extract a memory/knowledge claim, not the underlying event.
10. Evidence.answer must be an exact substring of the block's answer.
11. Do not classify contradictions.
12. Prior knowledge rules:
    - "never heard of X" -> relation="had_heard_of", object=X, negation=true.
    - "knew of X" -> relation="had_heard_of", object=X, negation=false.
    - "had mutual friends" about X supports relation="had_heard_of", object=X, negation=false.
    - "never met X face to face" is relation="met_face_to_face", not prior knowledge.

Output shape:
{{
  "blocks": [
    {{
      "block_id": "first-block-1",
      "claims": [
        {{
          "claim_id": "first_block1_claim1",
          "source": "{blocks[0].source}",
          "speaker": "witness",
          "topic": "short label summarizing the factual topic",
          "relation_family": "location_at_time",
          "standalone_claim": "The witness was at home on the evening of November 3rd.",
          "subject": "witness",
          "relation": "was_at",
          "object": "home",
          "time": {{
            "raw": "evening of November 3rd",
            "normalized24h": null,
            "minutes": null,
            "approximate": false
          }},
          "location": "home",
          "negation": false,
          "certainty": "certain",
          "question_context": {{
            "asked_about": "location",
            "time": "evening of November 3rd",
            "location": null
          }},
          "evidence": {{
            "question": "Where were you on the evening of November 3rd?",
            "answer": "exact quote from the answer"
          }}
        }}
      ]
    }}
  ]
}}

Allowed values:
- certainty: "certain", "uncertain", "denied", "unknown", or "does_not_remember"
- relation_family: "movement", "sleep_time", "ownership", "knowledge", "contact", "location_at_time", "action", or "unknown"
- time.normalized24h: "HH:MM", "24:00", or null
- time.minutes: integer from 0 to 1440, or null
- time.approximate: true or false

Time normalization rules:
- Use 24-hour time.
- "7 PM" -> normalized24h: "19:00", minutes: 1140
- "7:30 PM" -> normalized24h: "19:30", minutes: 1170
- "10 PM" -> normalized24h: "22:00", minutes: 1320
- "10:30 PM" -> normalized24h: "22:30", minutes: 1350
- "midnight" after evening/night context -> normalized24h: "24:00", minutes: 1440
- "12:00 AM" at the start of a day -> normalized24h: "00:00", minutes: 0
- If AM/PM is missing and context is unclear, keep raw but set normalized24h and minutes to null.
- If raw contains "around", "about", "maybe", "I think", mark approximate as true.
- Every time object must include raw, normalized24h, minutes, and approximate.

Q/A blocks:
{blocks_text}
""".strip()


def _parse_batch_json(text: str) -> list[dict[str, Any]]:
    json_text = _extract_json_object_text(text)
    if not json_text:
        raise ValueError("LLM response did not contain JSON.")

    parsed = json.loads(json_text, object_pairs_hook=_merge_duplicate_claims_keys)

    if isinstance(parsed, list):
        blocks = parsed
    elif isinstance(parsed, dict):
        blocks = parsed.get("blocks", [])
    else:
        blocks = []

    if not isinstance(blocks, list):
        raise ValueError("LLM JSON must contain a blocks array.")

    return [block for block in blocks if isinstance(block, dict)]


def _extract_json_object_text(text: str) -> str:
    trimmed = text.strip()

    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed

    start = trimmed.find("{")
    end = trimmed.rfind("}")

    if start < 0 or end < start:
        return ""

    return trimmed[start : end + 1]


def _merge_duplicate_claims_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """
    Ollama sometimes returns:
    {
      "claims": [...],
      "claims": [...],
      "claims": [...]
    }

    Normal json.loads keeps only the last key.
    This hook merges duplicate claims arrays instead.
    """
    result: dict[str, Any] = {}
    merged_claims: list[Any] = []

    for key, value in pairs:
        if key == "claims" and isinstance(value, list):
            merged_claims.extend(value)
            continue

        if key not in result:
            result[key] = value

    if merged_claims:
        result["claims"] = merged_claims

    return result


def _normalize_claims(
    raw_claims: list[dict[str, Any]],
    forced_source: ClaimSource | None = None,
    evidence_question: str | None = None,
    evidence_answer: str | None = None,
    block_id: str | None = None,
) -> list[ExtractedClaim]:
    claims: list[ExtractedClaim] = []

    for index, raw_claim in enumerate(raw_claims):
        normalized = _normalize_raw_claim(raw_claim, evidence_question or "", evidence_answer or "")
        if evidence_answer is not None:
            repaired_evidence = _evidence_from_answer(normalized["evidence"], evidence_answer)
            if repaired_evidence is None:
                debug_log(
                    "llm.claim_discarded",
                    {
                        "blockId": block_id,
                        "reason": "evidence_not_in_answer",
                        "evidence": normalized["evidence"],
                    },
                )
                continue
            normalized["evidence"] = repaired_evidence
            normalized["evidenceDetail"]["answer"] = repaired_evidence

        source: ClaimSource = forced_source or ("second" if normalized["source"] == "second" else "first")
        claim_id = _claim_id_for_block(source, block_id, len(claims) + 1)
        polarity = _normalize_polarity(normalized["polarity"], normalized)

        family = _normalize_family(normalized.get("relationFamily", ""))
        frame = assign_relation_family(normalized["relation"], family)

        claim = ExtractedClaim(
            id=claim_id,
            source=source,
            standaloneClaim=normalized["standaloneClaim"],
            speaker=normalized["speaker"],
            topic=normalized["topic"],
            subject=normalized["subject"],
            relation=frame.relation,
            relationFamily=frame.family,
            object=normalized["object"],
            polarity=polarity,
            negation=polarity == "negated",
            certainty=normalized["certainty"],
            questionContext=normalized["questionContext"],
            time=normalized["time"],
            location=normalized["location"] or None,
            uncertaintyMarkers=normalized["uncertaintyMarkers"],
            evidence=normalized["evidence"],
            evidenceDetail=normalized["evidenceDetail"],
        )

        if _is_usable_claim(claim):
            claims.append(claim)

    return claims


def _normalize_raw_claim(
    raw_claim: dict[str, Any],
    evidence_question: str = "",
    evidence_answer: str = "",
) -> dict[str, Any]:
    source = _clean(raw_claim.get("source")).lower()
    source = "second" if source == "second" else "first"

    topic = _clean(raw_claim.get("topic"))
    relation_family = _clean(raw_claim.get("relation_family") or raw_claim.get("relationFamily"))
    standalone_claim = _clean(raw_claim.get("standalone_claim") or raw_claim.get("standaloneClaim"))
    speaker = _clean(raw_claim.get("speaker")) or "witness"
    subject = _clean(raw_claim.get("subject"))
    relation = _clean(raw_claim.get("relation"))
    obj = _clean(raw_claim.get("object"))
    location = _clean(raw_claim.get("location"))
    evidence_detail = _normalize_evidence_detail(raw_claim.get("evidence") or raw_claim.get("Evidence"), evidence_question)
    evidence = _clean(evidence_detail["answer"])

    certainty = _normalize_certainty(raw_claim.get("certainty"))
    negation = _normalize_negation(raw_claim.get("negation"), raw_claim)
    raw_markers = raw_claim.get("uncertaintyMarkers") or raw_claim.get("uncertainty_markers") or []
    uncertainty_markers = [
        _clean(marker).lower()
        for marker in raw_markers
        if isinstance(marker, str) and _clean(marker)
    ]
    uncertainty_markers.extend(_certainty_markers(certainty))

    time_obj = raw_claim.get("time") if isinstance(raw_claim.get("time"), dict) else {}
    normalized_time = _normalize_time(time_obj, evidence, uncertainty_markers)
    question_context = _normalize_question_context(
        raw_claim.get("question_context") or raw_claim.get("questionContext"),
        evidence_question,
        normalized_time["raw"] if normalized_time else "",
        location,
    )

    if not standalone_claim:
        standalone_claim = _build_standalone_claim(subject, relation, obj, normalized_time, location, negation, evidence)

    return {
        "source": source,
        "standaloneClaim": standalone_claim,
        "speaker": speaker,
        "topic": topic,
        "subject": subject,
        "relation": relation,
        "object": obj,
        "relationFamily": relation_family,
        "polarity": raw_claim.get("polarity") or ("negated" if negation else "affirmed"),
        "negation": negation,
        "certainty": certainty,
        "questionContext": question_context,
        "time": normalized_time,
        "location": location,
        "uncertaintyMarkers": uncertainty_markers,
        "evidence": evidence,
        "evidenceDetail": evidence_detail,
    }


def _claim_id_for_block(source: ClaimSource, block_id: str | None, fallback_number: int) -> str:
    if block_id:
        match = re.search(r"block-(?P<number>\d+)$", block_id)
        if match:
            return f"{source}_block{match.group('number')}_claim{fallback_number}"
    return f"{source}_claim{fallback_number}"


def _normalize_family(value: str) -> str:
    valid = {
        "movement", "sleep_time", "ownership", "knowledge", "contact",
        "location_at_time", "action", "unknown",
    }
    text = value.strip().lower().replace("_", " ").replace(" ", "_")
    if text in valid:
        return text
    return "unknown"


def _normalize_evidence_detail(value: Any, fallback_question: str) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            "question": _clean(value.get("question")) or fallback_question,
            "answer": _clean(value.get("answer")),
        }

    return {
        "question": fallback_question,
        "answer": _clean(value),
    }


def _normalize_certainty(value: Any) -> str:
    text = _clean(value).lower().replace("-", "_")
    if text in {"certain", "uncertain", "denied", "unknown", "does_not_remember"}:
        return text
    if text in {"does not remember", "do_not_remember", "cannot remember", "can't remember"}:
        return "does_not_remember"
    if text in {"maybe", "might", "probably", "possibly", "not_sure", "not sure"}:
        return "uncertain"
    return "certain"


def _normalize_negation(value: Any, raw_claim: dict[str, Any]) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        text = _clean(value).lower()
        if text in {"true", "yes", "negated", "denied", "negative"}:
            return True
        if text in {"false", "no", "affirmed", "positive"}:
            return False

    text = _clean(
        " ".join(
            [
                _clean(raw_claim.get("relation")),
                _clean(raw_claim.get("object")),
                _evidence_answer_text(raw_claim.get("evidence") or raw_claim.get("Evidence")),
                _clean(raw_claim.get("polarity")),
            ]
        )
    ).lower()

    return any(marker in text for marker in ["never", "not", "don't", "do not", "didn't", "no,"])


def _evidence_answer_text(value: Any) -> str:
    if isinstance(value, dict):
        return _clean(value.get("answer"))
    return _clean(value)


def _certainty_markers(certainty: str) -> list[str]:
    if certainty == "uncertain":
        return ["uncertain"]
    if certainty == "unknown":
        return ["unknown"]
    if certainty == "does_not_remember":
        return ["does_not_remember"]
    if certainty == "denied":
        return ["denied"]
    return []


def _normalize_question_context(
    value: Any,
    fallback_question: str,
    fallback_time: str,
    fallback_location: str,
) -> dict[str, str | None]:
    context = value if isinstance(value, dict) else {}
    question = fallback_question.lower()
    asked_about = _clean(context.get("asked_about") or context.get("askedAbout"))

    if not asked_about:
        if question.startswith("where"):
            asked_about = "location"
        elif question.startswith("when") or "what time" in question:
            asked_about = "time"
        elif question.startswith("who"):
            asked_about = "person"
        elif question.startswith("did") or question.startswith("have") or question.startswith("had"):
            asked_about = "fact_confirmation"
        else:
            asked_about = "unknown"

    return {
        "asked_about": asked_about,
        "time": _clean(context.get("time")) or fallback_time or None,
        "location": _clean(context.get("location")) or fallback_location or None,
    }


def _build_standalone_claim(
    subject: str,
    relation: str,
    obj: str,
    time: dict[str, Any] | None,
    location: str,
    negation: bool,
    evidence: str,
) -> str:
    clean_subject = subject or "witness"
    relation_text = relation.replace("_", " ") or "stated"
    parts = ["The", clean_subject if clean_subject != "witness" else "witness"]

    if negation:
        parts.append("did not")

    parts.append(relation_text)
    if obj:
        parts.append(obj)
    if location and location != obj:
        parts.extend(["at", location])
    if time and time.get("raw"):
        parts.extend(["during", str(time["raw"])])

    sentence = " ".join(parts).strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence or evidence


def _normalize_time(
    time_obj: dict[str, Any],
    evidence: str,
    uncertainty_markers: list[str],
) -> dict[str, Any] | None:
    raw = _clean(time_obj.get("raw"))

    normalized24h = time_obj.get("normalized24h")
    normalized24h = normalized24h if isinstance(normalized24h, str) else None

    minutes = time_obj.get("minutes")
    minutes = minutes if isinstance(minutes, int) else None

    if minutes is None and normalized24h:
        parsed_from_normalized = _minutes_from_24h(normalized24h)
        if parsed_from_normalized is not None:
            minutes = parsed_from_normalized

    text_for_time = " ".join(part for part in [raw, evidence] if part)

    parsed_minutes, parsed_24h = _parse_time_phrase(text_for_time)

    if parsed_minutes is not None:
        minutes = parsed_minutes
        normalized24h = parsed_24h

    approximate = bool(time_obj.get("approximate"))
    lower_text = text_for_time.lower()

    if any(marker in lower_text for marker in ["around", "about", "maybe", "i think", "probably", "possibly"]):
        approximate = True

    if any(marker in {"around", "about", "maybe", "think", "probably", "possibly"} for marker in uncertainty_markers):
        approximate = True

    if not raw and minutes is None and normalized24h is None:
        return None

    return {
        "raw": raw,
        "minutes": minutes,
        "approximate": approximate,
    }


def _parse_time_phrase(text: str) -> tuple[int | None, str | None]:
    lower = text.lower()

    if "midnight" in lower:
        return 1440, "24:00"

    match = re.search(
        r"\b(?P<hour>1[0-2]|0?[1-9])(?::(?P<minute>[0-5][0-9]))?\s*(?P<ampm>a\.?m\.?|p\.?m\.?)\b",
        lower,
    )

    if match:
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
        ampm = match.group("ampm").replace(".", "")

        if ampm == "pm" and hour != 12:
            hour += 12

        if ampm == "am" and hour == 12:
            hour = 0

        minutes = hour * 60 + minute
        return minutes, _minutes_to_24h(minutes)

    # Handles phrases like "around 10, maybe 10:30" in evening deposition context.
    # We only infer PM for 7-11 because these examples are usually evening claims.
    vague_evening_match = re.search(
        r"\b(around|about|maybe)?\s*(?P<hour>7|8|9|10|11)(?::(?P<minute>[0-5][0-9]))?\b",
        lower,
    )

    if vague_evening_match and any(word in lower for word in ["evening", "night", "late", "sleep", "pm"]):
        hour = int(vague_evening_match.group("hour")) + 12
        minute = int(vague_evening_match.group("minute") or 0)
        minutes = hour * 60 + minute
        return minutes, _minutes_to_24h(minutes)

    return None, None


def _minutes_to_24h(minutes: int) -> str:
    if minutes == 1440:
        return "24:00"

    hour = minutes // 60
    minute = minutes % 60

    return f"{hour:02d}:{minute:02d}"


def _minutes_from_24h(value: str) -> int | None:
    match = re.match(r"^(?P<hour>2[0-4]|[01]?[0-9]):(?P<minute>[0-5][0-9])$", value.strip())
    if not match:
        return None

    hour = int(match.group("hour"))
    minute = int(match.group("minute"))

    if hour == 24 and minute != 0:
        return None

    return hour * 60 + minute



def _is_usable_claim(claim: ExtractedClaim) -> bool:
    if not claim.evidence:
        return False

    if not claim.relation:
        return False

    if not claim.subject and not claim.object and not claim.location:
        return False

    return True


def _normalize_polarity(value: Any, raw_claim: dict[str, Any] | None = None) -> ClaimPolarity:
    if raw_claim:
        if raw_claim.get("negation") is True:
            return "negated"
        if raw_claim.get("certainty") in {"unknown", "does_not_remember"} and raw_claim.get("relation") in {"remembered", "remember"}:
            return "negated"

    if isinstance(value, str):
        normalized = _clean(value).lower().replace("-", "_")

        affirmed = {
            "affirmed",
            "affirmative",
            "yes",
            "true",
            "positive",
            "asserted",
            "present",
            "known",
        }

        negated = {
            "negated",
            "negative",
            "denied",
            "deny",
            "denies",
            "no",
            "false",
            "not",
            "none",
            "never",
            "absent",
            "unknown_to_witness",
        }

        if normalized in affirmed:
            return "affirmed"

        if normalized in negated:
            return "negated"

    if raw_claim:
        text = _clean(
            " ".join(
                [
                    raw_claim.get("relation", ""),
                    raw_claim.get("object", ""),
                    raw_claim.get("evidence", ""),
                ]
            )
        ).lower()

        if any(marker in text for marker in ["never", "not", "don't", "do not", "didn't", "no,"]):
            return "negated"

    return "unknown"


def _clean(value: Any) -> str:
    if value is None:
        return ""

    if not isinstance(value, str):
        return ""

    return " ".join(value.split()).strip()


def _evidence_from_answer(evidence: str, answer: str) -> str | None:
    cleaned_evidence = _clean(evidence)
    cleaned_answer = _clean(answer)

    if not cleaned_evidence:
        return None

    if cleaned_evidence in cleaned_answer:
        return cleaned_evidence

    lower_answer = cleaned_answer.lower()
    lower_evidence = cleaned_evidence.lower()
    start = lower_answer.find(lower_evidence)

    if start >= 0:
        return cleaned_answer[start : start + len(cleaned_evidence)]

    return None


def _log_llm_output(
    provider: str,
    model: str,
    raw_text: str,
    raw_claims: list[dict[str, Any]],
    claims: list[ExtractedClaim],
    forced_source: ClaimSource | None = None,
    block_id: str | None = None,
) -> None:
    debug_log(
        "llm.output",
        {
            "provider": provider,
            "model": model,
            "forcedSource": forced_source,
            "blockId": block_id,
            "rawText": raw_text,
            "rawClaimCount": len(raw_claims),
            "normalizedClaimCount": len(claims),
        },
    )

    _log_triples(provider, model, claims)


def _log_triples(provider: str, model: str, claims: list[ExtractedClaim]) -> None:
    debug_log(
        "llm.triples",
        {
            "provider": provider,
            "model": model,
            "count": len(claims),
            "triples": [
                {
                    "id": claim.id,
                    "source": claim.source,
                    "standaloneClaim": claim.standalone_claim,
                    "speaker": claim.speaker,
                    "arg1": claim.subject,
                    "verb": claim.relation,
                    "relationFamily": claim.relation_family,
                    "arg2": claim.object,
                    "polarity": claim.polarity,
                    "negation": claim.negation,
                    "certainty": claim.certainty,
                    "questionContext": claim.question_context.model_dump(by_alias=True, exclude_none=True),
                    "evidence": claim.evidence,
                    "evidenceDetail": claim.evidence_detail.model_dump(exclude_none=True) if claim.evidence_detail else None,
                }
                for claim in claims
            ],
        },
    )



