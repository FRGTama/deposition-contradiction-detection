from __future__ import annotations

import os

import httpx


def provider_timeout() -> httpx.Timeout:
    seconds = _env_float("PROVIDER_TIMEOUT_SECONDS", 240)
    return httpx.Timeout(seconds, connect=30)


def timeout_error(provider: str, model: str) -> ValueError:
    seconds = _env_float("PROVIDER_TIMEOUT_SECONDS", 240)
    return ValueError(
        f"{provider} request timed out after {seconds:g}s while using model {model}. "
        "If this is a local Ollama model, make sure the model is pulled and warmed up, "
        "or increase PROVIDER_TIMEOUT_SECONDS."
    )


def provider_request_error(provider: str, model: str, error: Exception) -> ValueError:
    return ValueError(
        f"{provider} request failed while using model {model}: {error}. "
        "If this is Ollama, check that `ollama serve` is running, the base URL is correct, "
        "and the model is available locally."
    )


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default
