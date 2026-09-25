# 002 — EDGAR ingestion and parsing (P1)

**Status:** accepted · 2026-09-25

## Context
P2 onward needs 30 real 10-Ks (10 companies × 3 fiscal years), split into sections by Item number, with page and title metadata, so chunking can respect section boundaries and citations can point somewhere specific.

## Decisions
- **A plain retry loop over a library.** Rate limiting (fixed gap between requests) and retries (exponential backoff, honoring `Retry-After`, no retry on 4xx) are ~15 lines each. A library would hide exactly the logic I need to be able to explain.
- **Cache raw HTML forever, keyed by accession number.** A filing never changes once published (an amendment gets its own accession number), so the cache never goes stale. Re-running ingest costs ~10 requests (one submissions list per company) instead of ~40.
- **Two ways to detect a heading:** by Item number (`"Item 7."`, the normal case) and by section name for filings that use plain headings with no Item prefix. Both are anchored to the *start* of a text block, which is what keeps a cross-reference ("see Item 7 below") from being mistaken for a heading — the block starts with "see", not "Item".
- **One-content-row tables are heading candidates; multi-row tables are always data.** This was a real bug I found by running against live data, not something I anticipated (see below).
- **Keep the longest occurrence when an item's heading repeats.** Handles short "incorporated by reference" stubs correctly losing to the section with real body content.
- **Fail loudly per-company/per-filing, not for the whole run.** One bad company doesn't lose progress on the other nine; failures are collected and printed in the final summary instead of hidden or silently retried forever.

## The bug PLAN.md predicted, found in practice
Running against real SEC filings (not just the fixture), Amazon's 10-K was missing Item 7 entirely. The cause: Amazon lays out its real "Item 7." heading as a one-row, two-cell `<table>` — `"Item 7."` in one cell, the title in the other — used purely for column layout, not as a data table. My original code treated every `<table>` as data (converted to markdown, never checked for a heading), so that heading, and everything until the next successfully-matched heading, silently merged into the wrong section.

Fix: a table is only a heading candidate when it has exactly one non-empty content row (multi-row tables, like the table of contents, are always data — that's what keeps the TOC from generating a false heading per row). Confirmed the fix against all 30 real filings and added two tests for the shape.

This is exactly the risk PLAN.md's risk table named ("10-K HTML parsing is messy... spot-check 3 filings by eye"), which is why I ran against real data before calling the phase done instead of trusting the fixture alone.

## Alternative rejected
**`edgartools`** (PLAN.md's suggested fallback). Rejected for now because our own parser handles all 30 real filings correctly, and hand-rolling it means I can explain every heuristic and fix bugs like the one above myself. If a company outside our list of 10 breaks these heuristics badly, `edgartools` is the documented fallback.

## Most likely failure mode
**A layout trick we haven't seen yet.** SEC filing agents (Workiva, Donnelley, Toppan, and others) each have their own HTML quirks, and we've only verified 10 companies. A new company, or a company changing filing agents, could introduce a heading style neither of our two matching strategies catches — silently merging that section into its neighbor rather than erroring.

Mitigation already in place: the ingest summary always prints which items were found per filing, so a missing item shows up immediately rather than hiding in aggregate stats.

## Consequences
- `data/raw/` and `data/parsed/` are both gitignored (reproducible via `make ingest`); `data/manifest.json` is committed as the reproducibility record.
- P2's chunker can trust that `data/parsed/*.json` sections don't cross Item boundaries.
