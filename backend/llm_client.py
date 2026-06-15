from __future__ import annotations

import os

import httpx

from .http import provider_request_error, provider_timeout, timeout_error

_EXTRACTION_SYSTEM_PROMPT = (
    "You output strict JSON only. "
    "Return one object with one claims array. "
    "Do not repeat JSON keys."
)

_CLASSIFIER_SYSTEM_PROMPT = "You output strict JSON only."


async def llm_chat(
    prompt: str,
    *,
    system_prompt: str = _EXTRACTION_SYSTEM_PROMPT,
    max_tokens: int = 2400,
    label: str = "LLM",
) -> tuple[str, str]:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "anthropic":
        return await _call_anthropic(prompt, max_tokens, label)
    if provider == "openai":
        return await _call_openai(prompt, system_prompt, max_tokens, label)
    if provider == "deepseek":
        return await _call_deepseek(prompt, system_prompt, max_tokens, label)
    if provider == "ollama":
        return await _call_ollama(prompt, system_prompt, max_tokens, label)

    raise ValueError(
        f'Unsupported LLM_PROVIDER "{provider}". Use ollama, anthropic, openai, or deepseek.'
    )


async def _call_ollama(
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    label: str,
) -> tuple[str, str]:
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
                        "num_predict": max_tokens,
                    },
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error(f"Ollama {label}", model) from error
    except httpx.RequestError as error:
        raise provider_request_error(f"Ollama {label}", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"Ollama request failed: {response.status_code} {response.text}")

    return model, response.json().get("message", {}).get("content", "")


async def _call_anthropic(
    prompt: str,
    max_tokens: int,
    label: str,
) -> tuple[str, str]:
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
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error(f"Anthropic {label}", model) from error
    except httpx.RequestError as error:
        raise provider_request_error(f"Anthropic {label}", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"Anthropic request failed: {response.status_code} {response.text}")

    return model, (response.json().get("content") or [{}])[0].get("text", "")


async def _call_openai(
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    label: str,
) -> tuple[str, str]:
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
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error(f"OpenAI {label}", model) from error
    except httpx.RequestError as error:
        raise provider_request_error(f"OpenAI {label}", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"OpenAI request failed: {response.status_code} {response.text}")

    return model, (response.json().get("choices") or [{}])[0].get("message", {}).get("content", "")


async def _call_deepseek(
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    label: str,
) -> tuple[str, str]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
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
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
    except httpx.TimeoutException as error:
        raise timeout_error(f"DeepSeek {label}", model) from error
    except httpx.RequestError as error:
        raise provider_request_error(f"DeepSeek {label}", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"DeepSeek request failed: {response.status_code} {response.text}")

    return model, (response.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
