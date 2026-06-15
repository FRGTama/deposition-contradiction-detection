from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from .http import provider_request_error, provider_timeout, timeout_error
from .logger import debug_log
from .models import ClaimPolarity, ClaimSource, ExtractedClaim
from .qa import QABlock, split_qa_blocks
from .relation_frames import normalize_relation_frame


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

    if provider == "mock":
        claims = _mock_claims()
        _log_mock_claims_by_source(claims)
        return {
            "provider": provider,
            "model": "deterministic-fixture",
            "claims": claims,
            "blocks": first_blocks + second_blocks,
            "claimBlocks": {},
        }

    first = await _extract_claims_for_source(first_blocks, "first", provider)
    second = await _extract_claims_for_source(second_blocks, "second", provider)

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
    claims: list[ExtractedClaim] = []
    claim_blocks: dict[str, str] = {}
    model = ""

    for block in blocks:
        prompt = _build_extraction_prompt(block)

        if provider == "anthropic":
            raw = await _call_anthropic(prompt)
        elif provider == "openai":
            raw = await _call_openai(prompt)
        elif provider == "deepseek":
            raw = await _call_deepseek(prompt)
        elif provider == "ollama":
            raw = await _call_ollama(prompt)
        else:
            raise ValueError(
                f'Unsupported LLM_PROVIDER "{provider}". Use ollama, anthropic, openai, deepseek, or mock.'
            )

        block_claims = _normalize_claims(
            raw["raw_claims"],
            forced_source=source,
            id_start=len(claims) + 1,
            evidence_question=block.question,
            evidence_answer=block.answer,
            block_id=block.id,
        )
        for claim in block_claims:
            claim_blocks[claim.id] = block.id

        _log_llm_output(
            provider,
            raw["model"],
            raw["raw_text"],
            raw["raw_claims"],
            block_claims,
            source,
            block.id,
        )

        model = raw["model"]
        claims.extend(block_claims)

    return {"model": model, "claims": claims, "claimBlocks": claim_blocks}


def _build_extraction_prompt(block: QABlock) -> str:
    return f"""
You are an information extraction system for legal deposition transcripts.

Your task is to extract atomic factual claims from a single Q/A block.

Return valid JSON only. Return exactly one object with one top-level key: "claims".

Input metadata:
- source: "{block.source}"
- claim_id_prefix: "{block.source}_{block.id.replace('-', '_')}"

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
10. Evidence.answer must be an exact substring of the answer below.
11. Do not classify contradictions.
12. Prior knowledge rules:
    - "never heard of X" -> relation="had_heard_of", object=X, negation=true.
    - "knew of X" -> relation="had_heard_of", object=X, negation=false.
    - "had mutual friends" about X supports relation="had_heard_of", object=X, negation=false.
    - "never met X face to face" is relation="met_face_to_face", not prior knowledge.

Output shape:
{{
  "claims": [
    {{
      "claim_id": "{block.source}_{block.id.replace('-', '_')}_claim1",
      "source": "{block.source}",
      "speaker": "witness",
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
        "question": "{block.question}",
        "answer": "exact quote from the answer"
      }}
    }}
  ]
}}

Allowed values:
- certainty: "certain", "uncertain", "denied", "unknown", or "does_not_remember"
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

Q/A block:
Q: {block.question}
A: {block.answer}
""".strip()


async def _call_ollama(prompt: str) -> dict[str, Any]:
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    headers = {"Content-Type": "application/json"}

    if os.getenv("OLLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OLLAMA_API_KEY']}"

    try:
        async with httpx.AsyncClient(timeout=provider_timeout()) as client:
            response = await client.post(
                f"{base_url}/api/chat",
                headers=headers,
                json={
                    "model": model,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0,
                        "top_p": 0.1,
                        "num_predict": 2400,
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You output strict JSON only. "
                                "Return one object with one claims array. "
                                "Do not repeat JSON keys."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error("Ollama", model) from error
    except httpx.RequestError as error:
        raise provider_request_error("Ollama", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"Ollama request failed: {response.status_code} {response.text}")

    raw_text = response.json().get("message", {}).get("content", "")
    raw_claims = _parse_claims_json(raw_text)

    return {"model": model, "raw_text": raw_text, "raw_claims": raw_claims}


async def _call_anthropic(prompt: str) -> dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")

    try:
        async with httpx.AsyncClient(timeout=provider_timeout()) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": model,
                    "max_tokens": 2400,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error("Anthropic", model) from error
    except httpx.RequestError as error:
        raise provider_request_error("Anthropic", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"Anthropic request failed: {response.status_code} {response.text}")

    raw_text = (response.json().get("content") or [{}])[0].get("text", "")
    raw_claims = _parse_claims_json(raw_text)

    return {"model": model, "raw_text": raw_text, "raw_claims": raw_claims}


async def _call_openai(prompt: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

    try:
        async with httpx.AsyncClient(timeout=provider_timeout()) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You output strict JSON only. "
                                "Return one object with one claims array. "
                                "Do not repeat JSON keys."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error("OpenAI", model) from error
    except httpx.RequestError as error:
        raise provider_request_error("OpenAI", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"OpenAI request failed: {response.status_code} {response.text}")

    raw_text = (response.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
    raw_claims = _parse_claims_json(raw_text)

    return {"model": model, "raw_text": raw_text, "raw_claims": raw_claims}


async def _call_deepseek(prompt: str) -> dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek.")

    try:
        async with httpx.AsyncClient(timeout=provider_timeout()) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You output strict JSON only. "
                                "Return one object with one claims array. "
                                "Do not repeat JSON keys."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error("DeepSeek", model) from error
    except httpx.RequestError as error:
        raise provider_request_error("DeepSeek", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"DeepSeek request failed: {response.status_code} {response.text}")

    raw_text = (response.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
    raw_claims = _parse_claims_json(raw_text)

    return {"model": model, "raw_text": raw_text, "raw_claims": raw_claims}


def _parse_claims_json(text: str) -> list[dict[str, Any]]:
    json_text = _extract_json_object_text(text)

    if not json_text:
        raise ValueError("LLM response did not contain JSON.")

    parsed = json.loads(json_text, object_pairs_hook=_merge_duplicate_claims_keys)

    if isinstance(parsed, list):
        claims = parsed
    elif isinstance(parsed, dict):
        claims = parsed.get("claims")
    else:
        claims = None

    if not isinstance(claims, list):
        raise ValueError("LLM JSON must contain a claims array.")

    return [claim for claim in claims if isinstance(claim, dict)]


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
    id_start: int = 1,
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
        relation_frame = normalize_relation_frame(
            normalized["relation"],
            normalized["object"],
            normalized["evidence"],
            normalized["location"],
        )
        claim_id = _claim_id_for_block(source, block_id, len(claims) + 1)
        polarity = _normalize_polarity(normalized["polarity"], normalized)

        claim = ExtractedClaim(
            id=claim_id,
            source=source,
            standaloneClaim=normalized["standaloneClaim"],
            speaker=normalized["speaker"],
            topic=normalized["topic"],
            subject=normalized["subject"],
            relation=relation_frame.relation,
            relationFamily=relation_frame.family,
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

    if not subject:
        subject = _guess_subject_from_claim(topic, relation, obj, evidence)

    if not obj:
        obj = _guess_object_from_claim(relation, evidence, location)

    if not location:
        location = _guess_location_from_claim(relation, obj, evidence)

    if not topic:
        topic = _guess_topic_from_claim(relation, obj, location, evidence)

    if not relation:
        relation = _guess_relation_from_evidence(evidence)

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


def _guess_subject_from_claim(topic: str, relation: str, obj: str, evidence: str) -> str:
    text = " ".join([topic, relation, obj, evidence]).lower()

    if re.search(r"\b(i|me|my|we|witness|marcus webb)\b", text):
        return "witness"

    return ""


def _guess_object_from_claim(relation: str, evidence: str, location: str) -> str:
    lower_relation = relation.lower()
    lower_evidence = evidence.lower()

    if "daniel cho" in lower_evidence:
        return "Daniel Cho"

    if "victor lane" in lower_evidence:
        return "Victor Lane"

    if location and any(term in lower_relation for term in ["was at", "stayed", "located"]):
        return location

    if "hargrove street warehouse" in lower_evidence:
        return "Hargrove Street warehouse"

    if "hargrove street" in lower_evidence:
        return "Hargrove Street"

    if "home" in lower_evidence:
        return "home"

    if "groceries" in lower_evidence:
        return "groceries"

    if "pizza" in lower_evidence:
        return "pizza"

    if "restaurant" in lower_evidence:
        return "restaurant"

    if "pharmacy" in lower_evidence:
        return "pharmacy"

    if "phone call" in lower_evidence or "called" in lower_evidence:
        return "phone call"

    return ""


def _guess_location_from_claim(relation: str, obj: str, evidence: str) -> str:
    text = " ".join([relation, obj, evidence]).lower()

    if "home" in text:
        return "home"

    if "parking lot" in text:
        return "parking lot"

    if "hargrove street warehouse" in text:
        return "Hargrove Street warehouse"

    if "hargrove street" in text:
        return "Hargrove Street"

    return ""


def _guess_topic_from_claim(relation: str, obj: str, location: str, evidence: str) -> str:
    text = " ".join([relation, obj, location, evidence]).lower()

    if "victor lane" in text:
        return "Victor Lane prior knowledge"

    if any(term in text for term in ["daniel cho", "heard of", "knew of", "mutual friends"]):
        return "Daniel Cho prior knowledge"

    if any(term in text for term in ["sleep", "midnight"]):
        return "sleep time"

    if any(term in text for term in ["home", "apartment", "restaurant", "pharmacy", "parking lot", "warehouse", "hargrove", "went out", "went to", "left", "drove", "driven"]):
        return "location movement"

    if any(term in text for term in ["owned", "sold", "civic", "car"]):
        return "vehicle ownership"

    if any(term in text for term in ["waved", "seen", "neighbor", "tom", "met", "spoke", "phone call", "called"]):
        return "contact"

    return "general fact"


def _guess_relation_from_evidence(evidence: str) -> str:
    text = evidence.lower()

    if "met" in text and "face to face" in text:
        return "met_face_to_face"

    if "went to sleep" in text or "midnight" in text:
        return "went to sleep"

    if "went out" in text:
        return "went out"

    if "went to" in text:
        return "went to"

    if "was at home" in text:
        return "was at"

    if "was at my apartment" in text or "was at apartment" in text:
        return "was at"

    if "phone call" in text or "called" in text:
        return "called"

    if "ordered" in text:
        return "ordered"

    if "watched" in text:
        return "watched"

    if "sold" in text:
        return "sold"

    if "owned" in text or "did at the time" in text:
        return "owned"

    if "knew of" in text or "heard of" in text or "mutual friends" in text:
        return "had_heard_of"

    if "driven through" in text:
        return "driven through"

    if "waved" in text:
        return "waved"

    return ""


def _is_usable_claim(claim: ExtractedClaim) -> bool:
    if not claim.evidence:
        return False

    if not claim.topic:
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


def _log_mock_claims_by_source(claims: list[ExtractedClaim]) -> None:
    for source in ["first", "second"]:
        typed_source: ClaimSource = "second" if source == "second" else "first"
        source_claims = [claim for claim in claims if claim.source == source]
        raw_text = json.dumps(
            {
                "claims": [
                    claim.model_dump(by_alias=True, exclude_none=True)
                    for claim in source_claims
                ]
            }
        )
        _log_llm_output(
            "mock",
            "deterministic-fixture",
            raw_text,
            [
                claim.model_dump(by_alias=True, exclude_none=True)
                for claim in source_claims
            ],
            source_claims,
            typed_source,
        )


def _mock_claims() -> list[ExtractedClaim]:
    return _normalize_claims(
        [
            {
                "source": "first",
                "topic": "location evening november 3",
                "subject": "Marcus Webb",
                "relation": "stayed home all evening",
                "object": "home",
                "polarity": "affirmed",
                "location": "home",
                "evidence": "I was at home all evening.",
            },
            {
                "source": "first",
                "topic": "food evening november 3",
                "subject": "Marcus Webb",
                "relation": "ordered",
                "object": "pizza around 7pm",
                "polarity": "affirmed",
                "time": {
                    "raw": "around 7pm",
                    "normalized24h": "19:00",
                    "approximate": True,
                },
                "uncertaintyMarkers": ["around"],
                "evidence": "I ordered pizza around 7pm and watched TV.",
            },
            {
                "source": "first",
                "topic": "contact evening november 3",
                "subject": "Marcus Webb",
                "relation": "spoke to anyone",
                "object": "no one",
                "polarity": "negated",
                "evidence": "No, I was alone.",
            },
            {
                "source": "first",
                "topic": "sleep evening november 3",
                "subject": "Marcus Webb",
                "relation": "went to sleep",
                "object": "around 10 maybe 10:30",
                "polarity": "affirmed",
                "time": {
                    "raw": "Around 10, maybe 10:30",
                    "normalized24h": "22:00",
                    "approximate": True,
                },
                "uncertaintyMarkers": ["around", "maybe"],
                "evidence": "Around 10, maybe 10:30.",
            },
            {
                "source": "first",
                "topic": "warehouse prior visits",
                "subject": "Marcus Webb",
                "relation": "been to Hargrove Street warehouse",
                "object": "never",
                "polarity": "negated",
                "location": "Hargrove Street warehouse",
                "evidence": "No, never. I don't even know where that is.",
            },
            {
                "source": "first",
                "topic": "Daniel Cho prior knowledge",
                "subject": "Marcus Webb",
                "relation": "had_heard_of",
                "object": "Daniel Cho",
                "polarity": "negated",
                "evidence": "I'd never heard of him before this whole thing started.",
            },
            {
                "source": "second",
                "topic": "location evening november 3",
                "subject": "Marcus Webb",
                "relation": "went out briefly",
                "object": "groceries",
                "polarity": "affirmed",
                "time": {
                    "raw": "maybe around 7:30",
                    "normalized24h": "19:30",
                    "approximate": True,
                },
                "uncertaintyMarkers": ["think", "maybe", "around"],
                "evidence": "I think I went out briefly to get some groceries, maybe around 7:30.",
            },
            {
                "source": "second",
                "topic": "contact evening november 3",
                "subject": "Marcus Webb",
                "relation": "was seen by neighbor",
                "object": "waved in parking lot",
                "polarity": "affirmed",
                "uncertaintyMarkers": ["might"],
                "evidence": "My neighbor, Tom, might have seen me. We waved or something in the parking lot.",
            },
            {
                "source": "second",
                "topic": "sleep evening november 3",
                "subject": "Marcus Webb",
                "relation": "went to sleep",
                "object": "midnight maybe",
                "polarity": "affirmed",
                "time": {
                    "raw": "Midnight maybe",
                    "normalized24h": "24:00",
                    "approximate": True,
                },
                "uncertaintyMarkers": ["maybe"],
                "evidence": "It was late. Midnight maybe.",
            },
            {
                "source": "second",
                "topic": "warehouse prior visits",
                "subject": "Marcus Webb",
                "relation": "visited Hargrove Street area",
                "object": "driven through that part of town",
                "polarity": "affirmed",
                "location": "Hargrove Street area",
                "evidence": "I've driven through that part of town.",
            },
            {
                "source": "second",
                "topic": "Daniel Cho prior knowledge",
                "subject": "Marcus Webb",
                "relation": "had_heard_of",
                "object": "Daniel Cho",
                "polarity": "affirmed",
                "evidence": "I knew of him. We had mutual friends.",
            },
        ]
    )
