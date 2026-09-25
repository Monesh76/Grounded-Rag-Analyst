# CLAUDE.md — FilingsRAG

Question answering over SEC 10-K filings with grounded citations and an evaluation harness. Full plan in `PLAN.md`; work one phase at a time.

## Working agreement
- Work only on the phase I name. Propose a plan first; wait for approval before large changes.
- I'm building this for my portfolio and will be interviewed on it. Explain non-obvious choices briefly as you go, and prefer simple, readable code over clever code.
- Commit after each task with a clear message. Never commit secrets, `.env`, or `data/raw/`.
- If a requirement in PLAN.md seems wrong, say so instead of silently changing it.

## Stack
Python 3.12 · uv · FastAPI · Pydantic v2 · Postgres 16 + pgvector + full-text search · sentence-transformers (reranker, local embeddings) · DeepEval · Langfuse · Streamlit · Docker Compose · GitHub Actions.

## Commands
- `make setup` — install deps
- `docker compose up -d db` — start Postgres
- `make test` — unit tests (no network, no paid API calls)
- `make lint` — ruff check + format check
- `make ingest` / `make load` — fetch filings / chunk, embed, load
- `make serve` — run API locally
- `make eval` — retrieval-only evals (cheap, runs in CI)
- `make eval-full` — full evals with LLM judge (costs money; ask before running)

## Conventions
- Code lives in `src/filings_rag/`; tests mirror the package layout in `tests/`.
- Type hints everywhere; Pydantic models for anything crossing a boundary (API, DB rows, eval records).
- All config through `config.py` (pydantic-settings) reading `.env`. No model names, keys or thresholds hard-coded in logic.
- LLM and embedding access only through the `LLMProvider` / `Embedder` protocols.
- Unit tests must not hit the network or paid APIs; mock providers and use saved fixtures.
- SEC EDGAR requests use the configured User-Agent and stay under 8 requests/sec.
- Every answer must cite chunk ids as `[c:<id>]`; the unanswerable response is exactly `Not found in the filings.`

## Eval rules
- `evals/golden.jsonl` is hand-verified. Don't edit existing rows without asking; new drafted rows get `"verified": false`.
- Never lower a CI threshold to make a run pass. Report the regression and propose a fix instead.
- Eval results go to `results/` and are committed.
