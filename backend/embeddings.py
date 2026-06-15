from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

import httpx

from .http import provider_request_error, provider_timeout, timeout_error
from .logger import debug_log, text_preview, vector_preview
from .models import ExtractedClaim

LOCAL_VECTOR_SIZE = 384


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


class SimilarityIndex:
    def __init__(self, embeddings: list[ClaimEmbedding]) -> None:
        self.vectors = {embedding.claim_id: _normalize_vector(embedding.vector) for embedding in embeddings}

    def similarity(self, left_claim_id: str, right_claim_id: str) -> float:
        left = self.vectors.get(left_claim_id)
        right = self.vectors.get(right_claim_id)

        if left is None or right is None:
            return 0

        return _normalize_cosine(_cosine_similarity(left, right))

    def top_matches(self, left_claim_id: str, candidate_ids: list[str], limit: int) -> list[str]:
        return [
            candidate_id
            for candidate_id, _score in sorted(
                ((candidate_id, self.similarity(left_claim_id, candidate_id)) for candidate_id in candidate_ids),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
        ]


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

    if provider == "ollama":
        return await _embed_with_ollama(claims, extra_inputs)

    return _embed_locally(claims, provider, extra_inputs)


def create_similarity_index(embeddings: list[ClaimEmbedding]) -> SimilarityIndex:
    return SimilarityIndex(embeddings)


def embed_text_locally(text: str) -> list[float]:
    return _hash_text_to_vector(text)


def slot_embedding_id(claim_id: str, slot: Literal["subject", "relation", "object"]) -> str:
    return f"{claim_id}:{slot}"


def claim_to_embedding_text(claim: ExtractedClaim) -> str:
    return claim.standalone_claim.strip() or claim.evidence.strip()


def _claim_embedding_inputs(
    claims: list[ExtractedClaim],
    extra_inputs: list[EmbeddingInput] | None = None,
) -> list[EmbeddingInput]:
    inputs: list[EmbeddingInput] = []
    for claim in claims:
        inputs.extend(
            [
                EmbeddingInput(claim.id, claim_to_embedding_text(claim)),
                EmbeddingInput(slot_embedding_id(claim.id, "subject"), claim.subject),
                EmbeddingInput(slot_embedding_id(claim.id, "relation"), claim.relation),
                EmbeddingInput(slot_embedding_id(claim.id, "object"), claim.object),
            ]
        )
    if extra_inputs:
        inputs.extend(extra_inputs)
    return inputs


async def _embed_with_ollama(
    claims: list[ExtractedClaim],
    extra_inputs: list[EmbeddingInput] | None = None,
) -> EmbeddingResult:
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
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


def _embed_locally(
    claims: list[ExtractedClaim],
    provider: str,
    extra_inputs: list[EmbeddingInput] | None = None,
) -> EmbeddingResult:
    inputs = _claim_embedding_inputs(claims, extra_inputs)
    model = f"hashed-{LOCAL_VECTOR_SIZE}"
    embeddings = [
        ClaimEmbedding(embedding_input.id, _hash_text_to_vector(embedding_input.text))
        for embedding_input in inputs
    ]

    _log_embedding_inputs(provider, model, inputs)
    _log_embedding_vectors(provider, model, "local", embeddings)

    return EmbeddingResult(provider, model, embeddings)


def _hash_text_to_vector(text: str) -> list[float]:
    vector = [0.0] * LOCAL_VECTOR_SIZE
    tokens = [token for token in _split_tokens(text.lower()) if len(token) > 2]

    for token in tokens:
        index = _positive_hash(token) % LOCAL_VECTOR_SIZE
        sign = 1 if _positive_hash(f"{token}:sign") % 2 == 0 else -1
        vector[index] += sign

    return vector


def _split_tokens(text: str) -> list[str]:
    token = ""
    tokens: list[str] = []
    for character in text:
        if character.isalnum() or character == "'":
            token += character
        elif token:
            tokens.append(token)
            token = ""
    if token:
        tokens.append(token)
    return tokens


def _positive_hash(value: str) -> int:
    hash_value = 2166136261
    for character in value:
        hash_value ^= ord(character)
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return hash_value


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(min(len(left), len(right))))


def _normalize_cosine(value: float) -> float:
    return max(0, min(1, value))


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
    path: Literal["batch", "fallback", "local"],
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
