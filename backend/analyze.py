from __future__ import annotations

from .embeddings import embed_claims
from .llm import extract_claims
from .models import AnalysisResult
from .scoring import score_contradictions


async def analyze_depositions(transcript1: str, transcript2: str) -> AnalysisResult:
    transcript1 = transcript1.strip()
    transcript2 = transcript2.strip()

    if len(transcript1) == 0 or len(transcript2) == 0:
        raise ValueError("Both transcripts need enough text to compare.")

    extraction = await extract_claims(transcript1, transcript2)
    embedding_result = await embed_claims(extraction["claims"])
    contradictions = await score_contradictions(
        extraction["claims"],
        embedding_result.embeddings,
    )

    return AnalysisResult(
        claims=extraction["claims"],
        contradictions=contradictions,
        provider=extraction["provider"],
        model=extraction["model"],
        embeddingProvider=embedding_result.provider,
        embeddingModel=embedding_result.model,
    )
