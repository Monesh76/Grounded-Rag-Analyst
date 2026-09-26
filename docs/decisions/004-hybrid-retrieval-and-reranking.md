# 004 — Hybrid retrieval and reranking (P3)

**Status:** accepted · 2026-09-25

## Context
P2 leaves 6,513 embedded, full-text-indexed chunks in Postgres. P3 turns that into an actual retrieval path: dense search, keyword search, fused, reranked — and needs to prove, on real questions against real data, that the right section shows up in the top 6.

## Decisions
- **One `search()` entry point, four modes** (`dense | keyword | hybrid | hybrid_rerank`), so P6's experiment grid can compare them without touching calling code.
- **`reciprocal_rank_fusion` is a pure function over any hashable id**, unit-tested with hand-made string rankings (per PLAN.md), independent of the database; a thin wrapper maps fused ids back to full `RetrievalResult` objects.
- **Reranking runs over the full fused pool (up to top-50), not just the final top-6.** Reranking only within an already-cut-down list would defeat the point of scoring each candidate against the question directly.
- **Metadata filters** (company/ticker, fiscal year) are extracted from the question text and applied as a `WHERE` clause in both dense and keyword search, rather than relying on similarity alone to find the right company.
- **Local cross-encoder** (`BAAI/bge-reranker-base`) for reranking — no added API cost, and PLAN.md's suggested BGE family.

## Two real bugs, both only visible against live infrastructure
1. **pgvector's psycopg dumper doesn't cover a plain Python list.** It's registered for `pgvector.Vector` and `numpy.ndarray` — not `list`, which is exactly what both embedders return. `INSERT` worked anyway (pgvector defines an assignment cast from a float array to `vector`, used for column targets), but the `<=>` operator in a `SELECT` needs an actual vector operand, and operator resolution doesn't apply that cast. Fixed by wrapping the query embedding in `Vector(...)` before binding it.
2. **A far more serious one: `tests/test_db.py`'s fixture ran `DROP TABLE IF EXISTS chunks` against the same database `make load` populates.** Every full test-suite run since P2's real load had been silently destroying the 6,513 loaded rows — caught only because P3's acceptance check found an empty table where real data should have been. Fixed with a genuinely separate `filings_test` database (`Settings.test_database_url`, auto-created on first use), and every integration test fixture that mutates data now points at it. Verified by running the full suite twice and confirming via direct query that the real database survives. Real data was reloaded (`make load`, ~$0.65, ~19 min) before re-running the acceptance check.

The lesson from both: a green test suite proves the code *as tested*, not that it's safe to run against real state. Neither bug would have been caught without actually running against Postgres and real data and checking afterward, rather than trusting "tests passed."

## The actual acceptance check
Five questions, run for real via `python -m filings_rag.retrieve "..." --mode hybrid_rerank` against the live 6,513-chunk dataset:

| Question | Right section | Result |
|---|---|---|
| Apple's revenue in fiscal 2025 | Item 8 | ✅ in top 6 |
| NVIDIA's competition risks | Item 1A | ✅ dominates top 6 |
| JPMorgan's business overview | Item 1 | ✅ top 3 |
| Visa's net revenue in fiscal 2025 | Item 7 | ✅ rank 4, exact sentence |
| Apple's supply-chain risks | Item 1A | ✅ rank 3, exact topic |

All 5 pass, meeting PLAN.md's "Done when" for P3.

## A quality observation worth carrying into P5
NVIDIA's competition-risk query returned 4 near-duplicate chunks (identical reranker score, all about export controls) in a 6-result window, at the expense of topic diversity. Likely cause: chunk overlap producing repetitive text across consecutive chunks in a densely-worded risk section. Not a P3 blocker — the right *section* is still found — but worth watching once P5's eval harness measures answer quality, not just section-level recall.

## Alternative rejected
**Hosted rerank APIs** (e.g. Cohere Rerank). Rejected in favor of a local cross-encoder: no added per-query cost, and it's already the PLAN.md-suggested approach given the reranker only needs to score at query time, not embed a whole corpus.

## Most likely failure mode
**A question that names no company and no year.** Filter extraction returns `Filters(ticker=None, fiscal_year=None)`, so retrieval searches across all 10 companies × 3 years unfiltered — fine for genuinely cross-company questions, but a real risk for an ambiguous single-company question the extractor fails to match (e.g. a nickname or misspelling), where the answer could get diluted by a much larger unfiltered candidate pool.

## Consequences
- `db.get_test_connection()` / `Settings.test_database_url` are now the required pattern for any test that mutates chunk data; `test_db_smoke.py` is the one exception (read-only, no mutation).
- P4's `pipeline.ask()` will call `search()` directly rather than duplicating any of dense/keyword/fusion/rerank.
