from __future__ import annotations


def normalize_text(value: str) -> str:
    return (
        value.lower()
        .replace("_", " ")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("didn't", "did not")
        .replace("don't", "do not")
        .strip()
    )


def slugify(value: str) -> str:
    token = ""
    tokens: list[str] = []
    for character in normalize_text(value):
        if character.isalnum():
            token += character
        elif token:
            tokens.append(token)
            token = ""
    if token:
        tokens.append(token)
    return "_".join(tokens)


def clamp(value: float, minimum: float = 0, maximum: float = 1) -> float:
    return min(maximum, max(minimum, value))


def round_score(value: float) -> float:
    return round(value * 100) / 100
