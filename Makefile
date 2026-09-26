.PHONY: setup test lint format ingest load serve eval eval-full

setup:  ## Install Python 3.12 + all deps into .venv
	uv sync

test:  ## Run tests (the DB smoke test needs `docker compose up -d db`)
	uv run pytest

lint:  ## Lint and check formatting (no changes)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

ingest:  ## Download + parse the 10 companies' 10-Ks (needs SEC_USER_AGENT in .env)
	uv run python -m filings_rag.ingest

load:  ## Chunk, embed and load parsed filings into Postgres (needs `make ingest` first)
	uv run python -m filings_rag.ingest.load

serve:  ## Run the API locally at http://localhost:8000
	uv run uvicorn filings_rag.api:app --reload --port 8000

eval:  ## Retrieval-only evals (Recall@6, MRR) -- no LLM call, cheap, safe for CI
	uv run python -m evals.run_evals

eval-full:  ## Full pipeline evals with LLM judge -- costs money, ask before running
	uv run python -m evals.run_evals --full
