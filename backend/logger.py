from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any


def debug_log(event: str, payload: Any) -> None:
    if os.getenv("DEBUG_PIPELINE") != "true":
        return

    print(
        json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "payload": _json_safe(payload),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def text_preview(text: str, max_length: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[:max_length]}..."


def vector_preview(vector: list[float]) -> dict[str, Any]:
    norm = math.sqrt(sum(value * value for value in vector))
    return {
        "dimensions": len(vector),
        "norm": _round(norm, 4),
        "preview": [_round(value, 4) for value in vector[:8]],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(by_alias=True, exclude_none=True))
    return value


def _round(value: float, places: int) -> float:
    return round(value, places)
