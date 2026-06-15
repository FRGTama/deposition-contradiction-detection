from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from .http import provider_request_error, provider_timeout, timeout_error
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

    if provider == "mock":
        result = _mock_classification(claim1, claim2, semantic_score)
        _log_output(provider, "deterministic-fixture", result, "mock")
        return result

    prompt = _build_classifier_prompt(payload)

    if provider == "anthropic":
        model, raw_text = await _call_anthropic(prompt)
    elif provider == "openai":
        model, raw_text = await _call_openai(prompt)
    elif provider == "deepseek":
        model, raw_text = await _call_deepseek(prompt)
    elif provider == "ollama":
        model, raw_text = await _call_ollama(prompt)
    else:
        raise ValueError(f'Unsupported LLM_PROVIDER "{provider}" for classifier.')

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


async def _call_ollama(prompt: str) -> tuple[str, str]:
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
                    "options": {"temperature": 0, "top_p": 0.1, "num_predict": 900},
                    "messages": [
                        {"role": "system", "content": "You output strict JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error("Ollama classifier", model) from error
    except httpx.RequestError as error:
        raise provider_request_error("Ollama classifier", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"Ollama classifier request failed: {response.status_code} {response.text}")

    return model, response.json().get("message", {}).get("content", "")


async def _call_anthropic(prompt: str) -> tuple[str, str]:
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
                    "max_tokens": 900,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error("Anthropic classifier", model) from error
    except httpx.RequestError as error:
        raise provider_request_error("Anthropic classifier", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"Anthropic classifier request failed: {response.status_code} {response.text}")

    return model, (response.json().get("content") or [{}])[0].get("text", "")


async def _call_openai(prompt: str) -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

    try:
        async with httpx.AsyncClient(timeout=provider_timeout()) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "You output strict JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error("OpenAI classifier", model) from error
    except httpx.RequestError as error:
        raise provider_request_error("OpenAI classifier", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"OpenAI classifier request failed: {response.status_code} {response.text}")

    return model, (response.json().get("choices") or [{}])[0].get("message", {}).get("content", "")


async def _call_deepseek(prompt: str) -> tuple[str, str]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek.")

    try:
        async with httpx.AsyncClient(timeout=provider_timeout()) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "You output strict JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error("DeepSeek classifier", model) from error
    except httpx.RequestError as error:
        raise provider_request_error("DeepSeek classifier", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"DeepSeek classifier request failed: {response.status_code} {response.text}")

    return model, (response.json().get("choices") or [{}])[0].get("message", {}).get("content", "")


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


def _mock_classification(
    claim1: ExtractedClaim,
    claim2: ExtractedClaim,
    semantic_score: float,
) -> ClaimPairClassification:
    text = " ".join([
        claim1.topic,
        claim1.relation,
        claim1.object,
        claim1.location or "",
        claim1.evidence,
        claim2.topic,
        claim2.relation,
        claim2.object,
        claim2.location or "",
        claim2.evidence,
    ]).lower()

    if (
        claim1.relation == claim2.relation
        and _normalize_object(claim1.object) == _normalize_object(claim2.object)
        and {claim1.polarity, claim2.polarity} == {"affirmed", "negated"}
    ):
        return ClaimPairClassification(
            type="DIRECT",
            compatibility="incompatible",
            confidence=0.94,
            rationale="The claims make opposite polarity assertions about the same normalized fact.",
        )

    if ("all evening" in text or "whole night" in text) and any(
        term in text for term in ["went out", "stepped out", "left", "restaurant", "groceries", "pharmacy"]
    ):
        return ClaimPairClassification(
            type="INFERENTIAL",
            compatibility="incompatible",
            confidence=0.86,
            rationale="One claim places the witness at home for the whole evening while the other says the witness left during that period.",
        )

    if "sleep" in text and any(term in text for term in ["phone call", "called", "made a call"]):
        return ClaimPairClassification(
            type="INFERENTIAL",
            compatibility="incompatible",
            confidence=0.84,
            rationale="One claim says the witness was asleep while the other describes an action during that sleep period.",
        )

    if any(food in text for food in ["thai food", "pizza", "ordered"]) and any(place in text for place in ["pharmacy", "groceries", "went out"]):
        return ClaimPairClassification(
            type="FALSE_POSITIVE",
            compatibility="compatible",
            confidence=0.82,
            rationale="Ordering food and going out around a nearby time can both be true.",
        )

    return ClaimPairClassification(
        type="FALSE_POSITIVE",
        compatibility="uncertain" if semantic_score < 0.55 else "compatible",
        confidence=0.62,
        rationale="The pair is related, but the structured facts do not clearly conflict.",
    )


def _normalize_object(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())


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
