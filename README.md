# Deposition Contradiction Detector

Compares two deposition transcripts from the same witness to identify contradictory statements.

## Quick Start

```bash
python -m pip install -r requirements.txt
npm install
cp .env.example .env
npm run dev
```

Open `http://localhost:5173`. Paste two deposition transcripts and click **Analyze**.

## Configuration

Copy `.env.example` to `.env` and set your provider credentials.

### LLM Provider (claim extraction + contradiction classification)

```env
LLM_PROVIDER=ollama                    # ollama | anthropic | openai | deepseek
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
PROVIDER_TIMEOUT_SECONDS=240
```

### Embedding Provider (semantic similarity)

```env
EMBEDDING_PROVIDER=ollama              # ollama | openai
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### Debug

```env
DEBUG_PIPELINE=true                    # logs full pipeline JSON to stdout
```

### Hosted Providers

```env
# Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# DeepSeek
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

If using Ollama, run `ollama pull <model>` first. Increase `PROVIDER_TIMEOUT_SECONDS` if models are slow to load.

## Pipeline

```
Two transcripts
│
├─ 1. Q&A block splitting (regex Q:/A: markers)
│
├─ 2. Batch claim extraction (1 LLM call per deposition)
│     Extracts atomic claims with subject, relation, object,
│     time, location, negation, certainty, topic, relation_family,
│     and exact evidence quotes from the source text.
│
├─ 3. Embedding (Ollama or OpenAI, claim text only)
│
├─ 4. Candidate pair retrieval (in-memory vector store)
│     Cross-joins first-deposition claims against second-deposition.
│     Filters by semantic similarity + structural signals (shared
│     subject, entity, topic, relation family, time bucket, location).
│
├─ 5. Batch NLI scoring (cross-encoder/nli-deberta-v3-base)
│     Computes contradiction scores for all candidate pairs in
│     a single model pass. Guardrail-handled pairs skip NLI.
│
├─ 6. Contradiction scoring (per pair)
│     │
│     ├── Fast guardrails (deterministic, no LLM):
│     │   ├── same_fact + opposite polarity → DIRECT (0.94)
│     │   ├── same_fact + same polarity → FALSE_POSITIVE (0.24)
│     │   └── knowledge denial vs. presence assertion → INFERENTIAL (0.82)
│     │
│     ├── LLM classifier (runs only if no fast guardrail matched):
│     │   Returns type (DIRECT/INFERENTIAL/FALSE_POSITIVE),
│     │   compatibility, confidence, rationale.
│     │
│     ├── Post-LLM guardrails:
│     │   ├── same_fact + conflicting polarity → overrides to DIRECT
│     │   └── LLM claims contradiction but is uncertain → FALSE_POSITIVE
│     │
│     └── Confidence scoring (app-owned, deterministic):
│         final = 0.25 × semantic + 0.50 × NLI + 0.25 × structured
│         final *= (1 − 0.3 × uncertainty_penalty)
│         └── uncertainty_penalty counts hedges ("maybe", "around",
│             "I think"), negation strength, and approximate times
│
└─ 7. Rank, deduplicate, emit top 10
```

## Contradiction Types

| Label | Meaning | Example |
|---|---|---|
| `DIRECT` | Explicitly cannot both be true. Same fact, opposite polarity. | "I owned the car" vs "I never owned the car" |
| `INFERENTIAL` | Individually plausible but cannot coexist. | "I was home all evening" vs "I went out for groceries around 7:30" |
| `FALSE_POSITIVE` | Related claims that can both be true. | "I ordered pizza" vs "I watched TV" |

## Scoring Fields

Each contradiction result includes:

| Field | Description |
|---|---|
| `fr` | Factual relation score — how strongly the facts conflict |
| `fu` | Uncertainty disagreement — how different the certainty levels are |
| `confidence` | Combined score from semantic similarity, NLI, structured mismatch, and uncertainty penalty |
| `topicScore` | Semantic cosine similarity between claim embeddings |
| `type` | `DIRECT`, `INFERENTIAL`, or `FALSE_POSITIVE` |
| `severity` | `HIGH`, `MEDIUM`, or `LOW` |
| `rationale` | Human-readable explanation from guardrail or LLM classifier |

## Architecture

```
src/                    React + TypeScript frontend (Vite)
  App.tsx               Main component
  types.ts              TypeScript interfaces
  components/           Metric, ClaimBlock, ScoreBar, ScorePill
  utils.ts              countByType, labelForFilter

backend/                Python FastAPI backend
  main.py               POST /api/analyze endpoint
  analyze.py            Pipeline orchestrator
  llm.py                Batch claim extraction + normalization
  llm_client.py         Shared LLM HTTP client (all providers)
  llm_classifier.py     Contradiction classifier prompt + parsing
  embeddings.py         Text embedding (Ollama/OpenAI)
  vector_store.py       Candidate pair retrieval + claim framing
  scoring.py            Guardrails, NLI scoring, confidence computation
  relation_frames.py    Relation family taxonomy + classification
  qa.py                 Q&A block splitting
  models.py             Pydantic data models
  text_utils.py         normalize_text, slugify, clamp, round_score
  http.py               HTTP timeout/error helpers
  logger.py             Debug logging (gated by DEBUG_PIPELINE)
```

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Starts FastAPI (port 8000) + Vite (port 5173) |
| `npm run build` | Builds frontend to `dist/client/` |
| `npm start` | Production: FastAPI serves built frontend |
| `npm run typecheck` | TypeScript type checking |

## Design Decisions

- **Confidence is app-owned**: Never accepted from LLM output. Computed deterministically from semantic, NLI, structured mismatch, and uncertainty signals.
- **Batch extraction**: All Q&A blocks from one deposition sent in a single LLM call. Two depositions processed concurrently.
- **Batch NLI**: All candidate pairs scored in one cross-encoder pass. Guardrail-handled pairs skip inference.
- **Dual guardrail system**: Fast deterministic guardrails catch obvious same-fact reversals before the LLM classifier runs. Post-LLM guardrails override the classifier when structured evidence conflicts with its judgment.
- **LLM-classified relation families**: The extraction LLM outputs one of 8 relation families (movement, sleep_time, ownership, knowledge, contact, location_at_time, action, unknown) per claim. Used for candidate filtering, same-fact detection, and structured mismatch scoring.
- **Evidence anchoring**: Every claim must include an exact evidence substring from the transcript. Claims without verifiable evidence are discarded.
