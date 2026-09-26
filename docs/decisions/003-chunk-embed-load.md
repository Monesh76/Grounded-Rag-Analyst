# 003 — Chunk, embed, load (P2)

**Status:** accepted · 2026-09-25

## Context
P1 leaves us with 30 filings parsed into sections. P3's retrieval needs those sections split into retrieval-sized chunks, embedded, and searchable in Postgres by both vector similarity and full-text.

## Decisions
- **Section-aware chunking, greedy packing with paragraph/table-granularity overlap.** Never crosses a section boundary; tables are kept whole even when oversized (splitting one would produce orphaned rows with no header, worse than one long chunk); an oversized paragraph is split by sentence instead, since that's safe.
- **Token counts are a words×4/3 estimate, not a real tokenizer.** `tiktoken` needs a network call to fetch its encoding file on first use, which breaks "no network in unit tests." A rough estimate is enough for a chunk-size *boundary*; it doesn't need to match any specific embedding model's count.
- **Deterministic chunk ids** (`TICKER_FISCALYEAR_ITEM_INDEX`, e.g. `AAPL_2025_7_3`): citations are self-describing, and the loader is idempotent by construction — upsert by id, no separate dedup logic needed.
- **Embedder protocol, two implementations, one shared retry helper.** Extracted `http_retry.py`'s `RateLimiter`/`request_with_retry` out of `edgar.py` rather than duplicating it in the Voyage client — same shape (space out requests, retry with backoff honoring `Retry-After`, don't retry a 4xx).
- **Voyage AI (`voyage-finance-2`) chosen as production embedder** over OpenAI, since it's Anthropic's recommended embedding partner and finance-tuned, fitting this project's domain — your call, asked up front since it's a real cost/account decision.
- **`chunks` table sized for Voyage's 1024 dims.** pgvector needs a fixed dimension per column; the local embedder's 384-dim output would need its own column or table to compare side-by-side. Not built here since PLAN.md's schema literally has one `embedding` column — noted rather than silently expanded.
- **`embed.py` placed at `src/filings_rag/`, not under `ingest/`** (PLAN.md's file tree puts it under `ingest/`) — flagged rather than followed silently, because P3's retrieval will reuse the same `Embedder` to embed the user's question, not just for ingestion.

## Two real bugs found by running against a live database, not by reasoning
1. **A bare `conn.execute()` before a `with conn.transaction():` block silently breaks that block's commit.** The migration's DDL ran and was even queryable within the same session, but vanished on reconnect. Confirmed with a minimal repro against real Postgres — not documented behavior I'd have predicted. Compounded by a second issue: a multi-statement raw SQL string executed together with a later parameterized statement inside one transaction block also dropped part of what should have committed. Fixed by giving each migration file its own explicit `commit()` rather than relying on the context manager; safe because every statement in a migration uses `IF NOT EXISTS`, so re-running a partially-applied migration is harmless.
2. **`tests/test_db.py`'s own fixture wrapped each test in `with connection:`.** A test that intentionally triggers and catches a DB error (to prove pgvector rejects the wrong dimension) left the connection's transaction aborted; exiting that `with` block then rolled back *everything* the test had done, migrations included — which is why the `chunks` table disappeared after a full test run despite every individual test passing. Fixed by not wrapping the fixture in an implicit transaction at all.

Both were caught only because I ran the real migration against the real dev database and reconnected to check, rather than trusting "the test passed" — the tests were passing throughout because they read back state within the same still-open, doomed transaction.

## An account-level surprise, not a code bug
`make load` hit Voyage's `429`: the free tier without a payment method on file caps requests at 3/min and 10K tokens/min, well below what a 100-chunk batch needs. This is separate from — and hit long before — Voyage's free 200M-token allowance (this load used ~5.4M tokens, still $0 either way). You added a payment method to lift the rate cap; the request/payload code itself was already correct (verified via the raw error body, and again once the real 1024-dim vectors came back successfully).

## Alternative rejected
**Alembic** (or another migration framework) for the one-table schema. A dozen readable lines applying numbered `.sql` files and tracking them in `schema_migrations` covers this project's actual need, and I can explain and debug every line of it myself — which the transaction bug above required doing.

## Most likely failure mode
**A migration file that isn't idempotent.** The current design's safety net (separate commits, no atomicity between "applied" and "recorded") depends entirely on every migration statement using `IF NOT EXISTS`. A future migration that does something non-idempotent (e.g. `ALTER TABLE ... ADD COLUMN` without a guard) and crashes partway through would leave the schema in a partially-applied state that a naive re-run can't safely repair. Mitigation: keep that constraint explicit in `db.py`'s docstring (already done) and review new migrations against it.

## Consequences
- Real run: 6,513 chunks across 30 filings, 5,404,753 tokens, ~$0.65 (Voyage). Verified idempotent against live data (re-run produced the same 6,513, no duplication).
- Switching to the local embedder for a real load would need a new migration (separate `embedding_local vector(384)` column or table), not just a config flag — see the dimension note above.
