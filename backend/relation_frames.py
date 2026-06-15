from __future__ import annotations

import json
from dataclasses import dataclass

from .llm_client import llm_chat
from .models import RelationFamily
from .text_utils import slugify

_FAMILIES = [
    "movement",
    "sleep_time",
    "ownership",
    "knowledge",
    "contact",
    "location_at_time",
    "action",
    "unknown",
]


@dataclass(frozen=True)
class RelationFrame:
    relation: str
    family: RelationFamily
    functional: bool
    required_slots: tuple[str, ...]


FAMILY_CONSTRAINTS: dict[RelationFamily, dict[str, object]] = {
    "location_at_time": {
        "functional": True,
        "required_slots": ("subject", "location", "time"),
    },
    "movement": {
        "functional": False,
        "required_slots": ("subject", "object", "time"),
    },
    "sleep_time": {
        "functional": True,
        "required_slots": ("subject", "time"),
    },
    "ownership": {
        "functional": True,
        "required_slots": ("subject", "object"),
    },
    "knowledge": {
        "functional": False,
        "required_slots": ("subject", "object"),
    },
    "contact": {
        "functional": False,
        "required_slots": ("subject", "object"),
    },
    "action": {
        "functional": False,
        "required_slots": ("subject", "object"),
    },
    "unknown": {
        "functional": False,
        "required_slots": (),
    },
}


async def classify_relation_frames(items: list[dict]) -> list[RelationFamily]:
    if not items:
        return []

    prompt = _build_classify_prompt(items)
    _, raw_text = await llm_chat(prompt, max_tokens=400, label="relation-classifier")

    try:
        parsed = json.loads(_extract_json(raw_text))
        families = parsed.get("families") if isinstance(parsed, dict) else parsed
        if isinstance(families, list):
            result: list[RelationFamily] = []
            for entry in families[: len(items)]:
                family = _normalize_family(entry.get("family") if isinstance(entry, dict) else entry)
                result.append(family)
            while len(result) < len(items):
                result.append("unknown")
            return result
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return ["unknown"] * len(items)


def normalize_relation_frame(relation: str, obj: str = "", evidence: str = "", location: str = "") -> RelationFrame:
    return _frame(slugify(relation) or "unknown", "unknown")


def assign_relation_family(relation: str, family: RelationFamily) -> RelationFrame:
    return _frame(slugify(relation) or "unknown", family)


def relation_family(relation: str, obj: str = "", evidence: str = "", location: str = "") -> RelationFamily:
    return "unknown"


def is_functional_family(family: RelationFamily | str) -> bool:
    constraints = FAMILY_CONSTRAINTS.get(family)  # type: ignore[arg-type]
    return bool(constraints and constraints["functional"])


def families_are_comparable(family1: str, family2: str) -> bool:
    if family1 == "unknown" or family2 == "unknown":
        return True
    if family1 == family2:
        return True
    return {family1, family2} == {"location_at_time", "movement"}


def _frame(relation: str, family: RelationFamily) -> RelationFrame:
    constraints = FAMILY_CONSTRAINTS[family]
    return RelationFrame(
        relation=relation,
        family=family,
        functional=bool(constraints["functional"]),
        required_slots=tuple(constraints["required_slots"]),  # type: ignore[arg-type]
    )


def _build_classify_prompt(items: list[dict]) -> str:
    families_desc = (
        "movement: travel, go, leave, arrive, visit, drive, sail, fly, walk, return, enter, exit, head\n"
        "sleep_time: sleep, rest, nap, go to bed\n"
        "ownership: own, possess, have, acquire, buy, sell, belong\n"
        "knowledge: know, remember, hear of, recognize, learn, understand, forget, recall\n"
        "contact: meet, speak, call, greet, wave, text, email, communicate, interact\n"
        "location_at_time: be at, stay, remain, live, reside, inhabit, dwell, located\n"
        "action: order, watch, work, eat, drink, perform, make, create, other general actions\n"
        "unknown: cannot determine"
    )

    items_text = "\n".join(
        f"{i + 1}. relation=\"{item['relation']}\" object=\"{item.get('object', '')}\" "
        f"evidence=\"{item.get('evidence', '')[:120]}\" location=\"{item.get('location', '')}\""
        for i, item in enumerate(items)
    )

    return f"""Classify each relation into one of these families:

{families_desc}

For each item, choose the family that best describes the primary action or state expressed by the relation in the given context. Prefer a specific family over "action" or "unknown" when the relation clearly fits. Return valid JSON only.

Items:
{items_text}

Output: {{"families": [{{"family": "movement", "reason": "..."}}, ...]}}""".strip()


def _extract_json(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed
    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start < 0 or end < start:
        return "{}"
    return trimmed[start : end + 1]


def _normalize_family(value: str | None) -> RelationFamily:
    if not isinstance(value, str):
        return "unknown"
    text = value.strip().lower().replace("_", " ")
    for family in _FAMILIES:
        if family.replace("_", " ") == text:
            return family  # type: ignore[return-value]
    return "unknown"
