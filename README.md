# Deposition Contradiction Detector

Production-oriented take-home implementation for comparing two depositions from the same witness.

## Run

```bash
python -m pip install -r requirements.txt
npm install
npm run dev
```

Then open `http://localhost:5173`.

## Model Providers

The browser never calls an LLM directly. The server reads provider settings from `.env`.

```bash
cp .env.example .env
```

Default local setup:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
PROVIDER_TIMEOUT_SECONDS=240
WORDNET_AUTO_DOWNLOAD=true
```

Hosted alternatives:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
EMBEDDING_PROVIDER=local
```

For offline demos or tests:

```env
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=local
```

If Ollama times out on first use, run `ollama pull <model>` and try again, or increase `PROVIDER_TIMEOUT_SECONDS`.
WordNet is used as a best-effort lexical resource for verb opposition; set `WORDNET_AUTO_DOWNLOAD=false` to disable automatic corpus download.

## Architecture

- The backend is a FastAPI app that exposes `POST /api/analyze`.
- Vite proxies `/api` to FastAPI during local development.
- The LLM extracts normalized claims and evidence quotes.
- The app embeds each extracted claim and uses vector similarity to retrieve candidate pairs.
- A structured LLM classifier judges candidate-pair compatibility from normalized claim fields.
- The app computes final confidence with deterministic logic.
- Confidence is not accepted from model output.
- Direct, inferential, and false-positive cases are scored differently.

## Scripts

- `npm run dev`: starts FastAPI on `http://localhost:8000` and Vite on `http://localhost:5173`.
- `npm run build`: builds the React frontend into `dist/client`.
- `npm start`: starts FastAPI and serves the built frontend.

## Contradiction Scoring

Each claim pair receives:

- `fr`: factual relation contradiction score.
- `fu`: uncertainty disagreement score.
- `confidence`: app-owned score derived from `fr`, `fu`, semantic similarity, and evidence quality.

The final label is:

- `DIRECT`: explicit polarity or relation conflict.
- `INFERENTIAL`: individually plausible claims that cannot both hold.
- `FALSE_POSITIVE`: overlap with weak conflict or explainable imprecision.
