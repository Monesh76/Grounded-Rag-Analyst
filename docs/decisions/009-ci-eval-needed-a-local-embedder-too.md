# 009 — CI's free eval needed a local embedder, not just a pre-embedded corpus

**Status:** accepted · 2026-09-27

## Context
After pushing P6/P7's work, CI's `lint-and-test` job failed on `make eval-ci` with `ValueError: VOYAGE_API_KEY must be set in .env to use the Voyage embedder.` -- caught by the user watching the Actions run, not by anything in this session's own testing.

## What went wrong
`eval-ci` (decision record 007) was built around one correct fact -- the *corpus* is pre-embedded, so no re-embedding cost -- but missed a second one: dense retrieval also embeds the incoming *question* at query time, on every single call, regardless of whether the corpus already has vectors. CI has no `VOYAGE_API_KEY` by design (that was the whole point), so every `make eval-ci` run was going to fail this way from the moment it was written. This was never caught locally because local testing always ran against a `.env` with a real Voyage key already in it -- the exact same blind spot the P6/P7 decision records already named as a recurring failure mode ("verify by actually running it, not by reading it") reappeared here for the same underlying reason: the local dev environment quietly had something CI didn't.

Confirmed harmless in one respect: the failure costs $0. `VoyageEmbedder.from_settings()` raises before any network call, so nothing was ever charged -- but the check was broken from the first push, not a new regression.

## The fix
A new `chunks_local` table (migration `0003`), sized `vector(384)` for the local `sentence-transformers` embedder (`BAAI/bge-small-en-v1.5`) instead of Voyage's 1024-dim output. `tests/fixtures/eval_seed_local.sql` is the *same* 89 chunk rows as `eval_seed.sql`, re-embedded locally -- free, no API key, same underlying text. `experiments/configs/ci.yaml` points `make eval-ci` at `table: chunks_local` with `embedder_provider: local`, so both the corpus and the query embedding are local. `seed_eval_fixture.py` now loads both fixtures, so a real Voyage key still works locally against the same small dataset if someone wants to exercise the production embedder path.

Verified with `VOYAGE_API_KEY=""` explicitly cleared, not just unset from `.env` -- `make eval-ci` still passes (Recall@6 = MRR = 1.0), confirming the local path is genuinely independent of any Voyage credential, not silently falling back to it.

## Alternative rejected
**Switching `eval-ci`'s mode to `"keyword"`** (pure Postgres full-text search, no embedding call at all) was the cheapest possible fix -- no new table, no new fixture. Rejected: it would stop testing the actual `hybrid_rerank` code path CI is meant to guard, silently narrowing what "CI passes" means without telling anyone. The local-embedder fixture costs more to build but keeps `eval-ci` testing the real production retrieval shape.

## Consequences
- `CHUNK_TABLES`'s allow-list now has three entries (`chunks`, `chunks_fixed512`, `chunks_local`); any future addition follows the same pattern.
- The next time a config or fixture change touches embedding, the checklist is: does this path call `embedder.embed()` on a live input (a question), not just on a corpus at load time? That's the exact distinction this bug turned on.
