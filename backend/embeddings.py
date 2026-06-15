from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import httpx

from .http import provider_request_error, provider_timeout, timeout_error
from .logger import debug_log, text_preview, vector_preview
from .models import ExtractedClaim


@dataclass
class ClaimEmbedding:
    claim_id: str
    vector: list[float]


@dataclass
class EmbeddingInput:
    id: str
    text: str


@dataclass
class EmbeddingResult:
    provider: str
    model: str
    embeddings: list[ClaimEmbedding]


async def embed_claims(
    claims: list[ExtractedClaim],
    extra_inputs: list[EmbeddingInput] | None = None,
) -> EmbeddingResult:
    provider = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "ollama")).lower()
    debug_log(
        "config.embeddings",
        {
            "provider": provider,
            "ollamaEmbeddingModel": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            "openaiEmbeddingModel": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        },
    )

    if provider == "openai":
        return await _embed_with_openai(claims, extra_inputs)

    return await _embed_with_ollama(claims, extra_inputs)


def claim_to_embedding_text(claim: ExtractedClaim) -> str:
    return claim.standalone_claim.strip() or claim.evidence.strip()


def _claim_embedding_inputs(
    claims: list[ExtractedClaim],
    extra_inputs: list[EmbeddingInput] | None = None,
) -> list[EmbeddingInput]:
    inputs = [EmbeddingInput(claim.id, claim_to_embedding_text(claim)) for claim in claims]
    if extra_inputs:
        inputs.extend(extra_inputs)
    return inputs


async def _embed_with_ollama(
    claims: list[ExtractedClaim],
    extra_inputs: list[EmbeddingInput] | None = None,
) -> EmbeddingResult:
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    headers = {"Content-Type": "application/json"}
    inputs = _claim_embedding_inputs(claims, extra_inputs)

    if os.getenv("OLLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OLLAMA_API_KEY']}"

    _log_embedding_inputs("ollama", model, inputs)

    try:
        async with httpx.AsyncClient(timeout=provider_timeout()) as client:
            embed_response = await client.post(
                f"{base_url}/api/embed",
                headers=headers,
                json={"model": model, "input": [embedding_input.text for embedding_input in inputs]},
            )
    except httpx.TimeoutException as error:
        raise timeout_error("Ollama embedding", model) from error
    except httpx.RequestError as error:
        raise provider_request_error("Ollama embedding", model, error) from error

    if embed_response.status_code < 400:
        vectors = embed_response.json().get("embeddings") or []
        if len(vectors) != len(inputs):
            debug_log(
                "embedding.batch_mismatch",
                {
                    "provider": "ollama",
                    "model": model,
                    "expected": len(inputs),
                    "received": len(vectors),
                    "message": "Batch /api/embed returned mismatched vector count; falling back to per-item /api/embeddings.",
                },
            )
        if len(vectors) == len(inputs):
            embeddings = [
                ClaimEmbedding(embedding_input.id, [float(value) for value in vectors[index]])
                for index, embedding_input in enumerate(inputs)
            ]
            _log_embedding_vectors("ollama", model, "batch", embeddings)
            return EmbeddingResult("ollama", model, embeddings)

    embeddings: list[ClaimEmbedding] = []
    try:
        async with httpx.AsyncClient(timeout=provider_timeout()) as client:
            for embedding_input in inputs:
                response = await client.post(
                    f"{base_url}/api/embeddings",
                    headers=headers,
                    json={"model": model, "prompt": embedding_input.text},
                )

                if response.status_code >= 400:
                    raise ValueError(f"Ollama embedding request failed: {response.status_code} {response.text}")

                vector = response.json().get("embedding")
                if not vector:
                    raise ValueError("Ollama embedding response did not contain an embedding vector.")

                embeddings.append(ClaimEmbedding(embedding_input.id, [float(value) for value in vector]))
    except httpx.TimeoutException as error:
        raise timeout_error("Ollama embedding", model) from error
    except httpx.RequestError as error:
        raise provider_request_error("Ollama embedding", model, error) from error

    _log_embedding_vectors("ollama", model, "fallback", embeddings)
    return EmbeddingResult("ollama", model, embeddings)


async def _embed_with_openai(
    claims: list[ExtractedClaim],
    extra_inputs: list[EmbeddingInput] | None = None,
) -> EmbeddingResult:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    inputs = _claim_embedding_inputs(claims, extra_inputs)

    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")

    _log_embedding_inputs("openai", model, inputs)

    try:
        async with httpx.AsyncClient(timeout=provider_timeout()) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={"model": model, "input": [embedding_input.text for embedding_input in inputs]},
            )
    except httpx.TimeoutException as error:
        raise timeout_error("OpenAI embedding", model) from error
    except httpx.RequestError as error:
        raise provider_request_error("OpenAI embedding", model, error) from error

    if response.status_code >= 400:
        raise ValueError(f"OpenAI embedding request failed: {response.status_code} {response.text}")

    vectors = [item.get("embedding") or [] for item in response.json().get("data", [])]
    if len(vectors) != len(inputs) or any(not vector for vector in vectors):
        raise ValueError("OpenAI embedding response did not include one vector per embedding input.")

    embeddings = [
        ClaimEmbedding(embedding_input.id, [float(value) for value in vectors[index]])
        for index, embedding_input in enumerate(inputs)
    ]
    _log_embedding_vectors("openai", model, "batch", embeddings)
    return EmbeddingResult("openai", model, embeddings)


def _log_embedding_inputs(provider: str, model: str, inputs: list[EmbeddingInput]) -> None:
    debug_log(
        "embedding.inputs",
        {
            "provider": provider,
            "model": model,
            "count": len(inputs),
            "inputs": [{"id": embedding_input.id, "textPreview": text_preview(embedding_input.text)} for embedding_input in inputs],
        },
    )


def _log_embedding_vectors(
    provider: str,
    model: str,
    path: Literal["batch", "fallback"],
    embeddings: list[ClaimEmbedding],
) -> None:
    debug_log(
        "embedding.vectors",
        {
            "provider": provider,
            "model": model,
            "path": path,
            "count": len(embeddings),
            "vectors": [{"id": embedding.claim_id, **vector_preview(embedding.vector)} for embedding in embeddings],
        },
    )
