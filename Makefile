.PHONY: setup test lint format ingest

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
