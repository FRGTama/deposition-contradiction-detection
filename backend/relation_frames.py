from __future__ import annotations

import re
from dataclasses import dataclass

from .models import RelationFamily


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


def normalize_relation_frame(relation: str, obj: str = "", evidence: str = "", location: str = "") -> RelationFrame:
    text = _normalize(" ".join([relation, obj, evidence, location]))

    if _contains_any(text, ["sleep", "slept", "went to sleep", "bed", "midnight"]):
        return _frame("went_to_sleep", "sleep_time")

    if _contains_any(text, ["went out", "go out", "left", "leave", "stepped out", "went to", "go to", "restaurant", "pharmacy"]):
        return _frame("went_out", "movement")

    if _contains_any(text, ["drive", "drove", "driven", "arrive", "arrived", "visited", "visit", "been to", "warehouse", "hargrove"]):
        return _frame("traveled_to", "movement")

    if _contains_any(text, ["sold", "sell", "transferred", "gave"]):
        return _frame("transferred_ownership", "action")

    if _contains_any(text, ["own", "owned", "possess", "possessed", "belong", "civic", "car"]):
        return _frame("owned", "ownership")

    if _contains_any(text, ["never heard", "had heard of", "heard of", "knew of", "know of", "aware", "recognize", "recognized", "mutual friends"]):
        return _frame("had_heard_of", "knowledge")

    if _contains_any(text, ["met face to face", "meet face to face", "face to face"]):
        return _frame("met_face_to_face", "contact")

    if _contains_any(text, ["met face", "meet face", "met", "meet"]):
        return _frame("met", "contact")

    if _contains_any(text, ["spoke", "speak", "called", "call", "phone call", "text", "contact", "waved", "seen", "saw", "neighbor"]):
        return _frame("had_contact", "contact")

    if _contains_any(text, ["was at", "were at", "stayed", "remained", "located", "at home", "at apartment", "home all evening", "all evening", "whole night"]):
        return _frame("was_at", "location_at_time")

    if _contains_any(text, ["ordered", "watched", "worked", "bought", "purchased"]):
        return _frame(_snake_relation(relation) or "performed_action", "action")

    return _frame(_snake_relation(relation) or "unknown", "unknown")


def relation_family(relation: str, obj: str = "", evidence: str = "", location: str = "") -> RelationFamily:
    return normalize_relation_frame(relation, obj, evidence, location).family


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


def _contains_any(text: str, needles: list[str]) -> bool:
    for needle in needles:
        normalized = _normalize(needle)
        if " " in normalized:
            if normalized in text:
                return True
            continue

        if re.search(rf"\b{re.escape(normalized)}\b", text):
            return True

    return False


def _snake_relation(value: str) -> str:
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

    return "_".join(tokens[:4])


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
