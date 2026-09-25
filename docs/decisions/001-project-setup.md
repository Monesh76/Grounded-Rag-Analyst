# 001 — Project setup (P0)

**Status:** accepted · 2026-09-25

## Context
Every later phase needs a reproducible Python environment, one config source, a database that supports both vector and keyword search, and CI that runs on every change.

## Decisions
- **uv + committed `uv.lock`; CI installs with `uv sync --locked`.** CI gets exactly the versions tested locally, and fails if the lockfile is stale.
- **src layout (`src/filings_rag/`).** Tests import the installed package, not files that happen to be on the path, so packaging mistakes show up early.
- **One `Settings` class (pydantic-settings), cached by `get_settings()`.** Precedence: environment variable > `.env` > default. `extra="ignore"` because `.env` also holds `POSTGRES_*` vars that only docker-compose reads.
- **Postgres 16 + pgvector (`pgvector/pgvector:pg16`) as the only datastore.** Dense search (pgvector) and keyword search (`tsvector`) live in the same table, so hybrid retrieval and metadata filters are plain SQL.
- **Compose healthcheck via `pg_isready`;** CI uses the same image as a service container.
- **The DB smoke test fails rather than skips when Postgres is unreachable.** It runs `CREATE EXTENSION IF NOT EXISTS vector`, which is idempotent, then checks a known L2 distance.

## Alternative rejected
**A dedicated vector database (e.g. Qdrant or Pinecone) next to Postgres.** Rejected because:
- hybrid search would need two systems, plus fusion across network calls;
- filters (company, year) would have to be kept in sync in two places;
- it adds one more service to run and deploy.

At ~20–40K chunks, pgvector's HNSW index is well within its comfort zone. Revisit at tens of millions of vectors, or if we need features Postgres lacks.

## Most likely failure mode
**A port 5432 conflict with a Postgres already running on the host** (e.g. a Homebrew install). The smoke test then connects to that server instead of the container, and fails with `extension "vector" is not available`, which looks like a pgvector problem but isn't.

Fix: set `POSTGRES_PORT` (e.g. 5433) and the port in `DATABASE_URL` in `.env`.

## Consequences
- `make test` needs `docker compose up -d db`.
- `pytest -m "not integration"` runs the pure unit tests without Docker.
- P2's migration will take over creating the extension.
