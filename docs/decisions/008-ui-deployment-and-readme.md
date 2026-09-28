# 008 — UI, deployment and README (P7)

**Status:** accepted · 2026-09-27

## Context
P7 is the last engineering phase: a Streamlit UI, a production Dockerfile, deployment notes, and the README a stranger reads first. Like every phase since P4, running things for real (a browser against the live UI, a built Docker image against the live DB) surfaced problems that reading the code would not have.

## Decisions
- **Explicit `Filters` passthrough (UI dropdowns -> API -> `search()`), not question-text stuffing.** `search()`, `ask()`/`ask_detailed()`, and the `/ask` endpoint all take an optional `Filters` that wins over `extract_filters(question)`'s auto-detection when given. The alternative (prepending "For Apple in 2025:" to the question text and relying on the existing regex-based extractor) would have worked but made the UI's filters a fragile side effect of prompt engineering rather than a real parameter.
- **`Source` now carries the chunk's own `text`, not just its metadata.** The API previously had no way to return the passage a citation expands to -- PLAN.md's "citations that expand to show the exact passage" requirement needed this field to exist at all.
- **The `web-design-guidelines` skill does not apply to this UI**, and this was decided explicitly rather than silently ignored. Its own Section 13 states it is not for "dashboards / dense product UI / admin panels" and is built entirely around React/Next.js/Tailwind/Motion/GSAP -- none of which apply to a Streamlit app. What did transfer: one accent color (`.streamlit/config.toml`), avoiding filler/AI-tell copy, and WCAG-sane contrast -- verified by actually loading the rendered page, not assumed.
- **Cheap deployment (Streamlit Community Cloud + Cloud Run + Neon/Supabase, ~$0/month) is the documented default; Cloud Run + managed Postgres (PLAN.md's original suggestion) is kept as a documented alternative, not run.** The user's explicit choice, after being shown the real cost tradeoff (Cloud SQL doesn't scale to zero; a managed Postgres alone runs $7-10+/month regardless of traffic).
- **The Dockerfile builds the API only.** The UI is deployed separately on Streamlit Community Cloud, which runs `ui/app.py` directly from the repo without a Dockerfile -- building one for the UI would be dead weight for the chosen deployment path.

## Bugs found by actually running things, not by reading them
1. **The Streamlit UI's filter dropdowns and citation expander were verified with a real browser (`playwright-cli`) against the real running API and database**, not just written and assumed correct: selected a company and fiscal year, asked a real question, confirmed the citation rendered, the source expander opened to the exact retrieved passage, and the cost/latency/token caption showed real numbers ($0.0064, 30247ms, 6218 tokens) matching what the API actually returned.
2. **The Docker image's non-root user had no home directory**, and `sentence-transformers`/`huggingface_hub` cache the reranker model under `$HOME/.cache` on first use -- the container crashed with `PermissionError: [Errno 13] Permission denied: '/home/app'` on its first real request. Fixed with `useradd --create-home`.
3. **`docker run --env-file .env` silently corrupted every quoted value.** `.env`'s values are quoted (`VOYAGE_API_KEY="pa-..."`), which `pydantic-settings`' own parser strips correctly -- but Docker's `--env-file` does not strip quotes, passing the literal value including the quote characters. This broke the Voyage API key (401 Unauthorized) and the Langfuse base URL (DNS resolution failure on a hostname that literally started with a `"` character) simultaneously, on the exact same underlying cause. Not a bug in this project's code -- documented as a real gotcha for local Docker testing, since it doesn't affect the recommended deployment paths (Cloud Run secrets and Streamlit Cloud's settings UI both take unquoted values directly).
4. **`.env.example` was stale since P0**, missing every key added from P2 onward (Voyage, all three LLM providers, Langfuse). A stranger following the README's quickstart could not have actually configured `.env` from it. Fixed as part of this phase rather than left for later, since the README's whole point is that a stranger can follow it.

## Alternative rejected
**A custom HTML/React frontend instead of Streamlit.** PLAN.md specifies Streamlit explicitly, and the `web-design-guidelines` skill's own scope rules confirm it: this UI is closer to a dense internal tool (question box, filters, an expandable sources panel) than a marketing surface, which is exactly the category that skill defers elsewhere. Streamlit's tradeoffs (limited layout control, no custom component library) are accepted rather than fought.

## Most likely failure mode
**A future contributor testing the Docker image locally hits the same quoted-`.env` gotcha again**, since `docker run --env-file .env` is the first thing most people try. Documented in `docs/deployment.md`, not fixed at the tooling level (no `.env` quoting convention change, since `pydantic-settings` handles the quoted form correctly and changing it would be a larger, unrelated convention change for the whole project).

## Consequences
- The 3-command quickstart in the README reuses `tests/fixtures/eval_seed.sql` (P6's CI fixture) rather than inventing a second fixture -- one less thing to keep in sync.
- `filings_rag.migrate` (a new small CLI) is now the standalone way to apply migrations against a fresh database (a newly provisioned Neon/Supabase/Cloud SQL instance); every prior call site built its own connection and called `run_migrations()` inline, with no standalone entrypoint.
