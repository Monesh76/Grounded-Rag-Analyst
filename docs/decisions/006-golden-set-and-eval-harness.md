# 006 — Golden set and eval harness (P5)

**Status:** accepted · 2026-09-26

## Context
P4 leaves a working, citation-validated pipeline. P5 needs a hand-verified golden set and an eval harness that actually gates quality — deterministic retrieval metrics cheap enough for every PR, and a full-pipeline run with an LLM judge for the metrics that need one. This phase surfaced more real, load-bearing bugs than any prior one, entirely because it's the first thing in this project that runs the *whole* system, concurrently, against real data, for real money.

## Decisions
- **Golden set: 50 rows, drafted from the real parsed filings, all read from `data/parsed/*.json` directly** rather than guessed — matches PLAN.md §5's type mix exactly (single_fact 15, numeric_table 10, risk_narrative 8, comparison 7, unanswerable 10). All 50 are now hand-verified (`"verified": true`) by the user, meeting PLAN.md's actual P5 "Done when" in full, not just the code.
- **`make eval` (Recall@6, MRR only) vs `make eval-full` (real pipeline + judge) split by what needs an LLM call**, not by metric category — Recall@6/MRR need only retrieval, so they're free and CI-safe; citation precision and refusal accuracy are deterministic but need a real generated answer, so they live in eval-full alongside the judge metrics.
- **`citation_precision` is measured against the model's raw, pre-validation citations** (`ask_detailed()` exposes this), not the final `Answer` — otherwise the metric would tautologically always read 1.0, since the pipeline never lets an invalid citation reach the user.
- **The judge is Claude Sonnet 5, direct API** — a different vendor/model than whichever provider generated the answer, to avoid self-favoring bias.
- **Errors are caught per-row, never let one question's failure lose the whole run's results.**

## The bugs, in the order they were found
This phase found five real, previously-invisible problems, each only visible by actually running the system under conditions no earlier phase had exercised (concurrency, the full 50-question set, real judge calls, real cost).

1. **A parser bug, found while drafting the golden set from real data, not by testing the parser.** BAC's and GS's Item 7 was nearly empty; their real MD&A content had been misfiled under neighboring items. Root cause: both filers repeat the section heading as a running header on every page, and each repeat was treated as a brand-new section, fragmenting one long section into dozens of one-page candidates that the P1 dedup logic (built for a different scenario) discarded down to one page. Fixed by treating a repeat of the *exact same heading text* as a page continuation. Re-ran ingest and reload on the corrected data before drafting the golden set on top of it — see the `fix:` commit for full detail.
2. **A segfault from concurrent CPU-bound model calls.** `make eval`'s first real run crashed with concurrency=4: multiple threads calling the shared reranker's `CrossEncoder.predict()` concurrently caused severe thrashing (and once, an outright crash), even after capping torch's internal thread count. Fixed with a lock serializing local-model calls across workers, while each worker's LLM call — the actual bottleneck in eval-full — still runs concurrently outside that lock.
3. **A single API failure losing an entire run's results.** `eval-full`'s first real run hit OpenRouter's account-level "in-flight budget" limit on one request, and `ThreadPoolExecutor.map()`'s first-exception-wins behavior aborted the whole run with nothing written to disk. Fixed by catching per-row and reporting an error marker instead of raising.
4. **The judge's structured JSON output silently truncated, twice.** DeepEval's `FaithfulnessMetric` extracts claims and truths as JSON from the *full* retrieved context; sharing `llm_max_tokens` (1024, sized for a concise answer) truncated that mid-JSON against real chunk lengths. Fixed once to 4096 (still failed on a 6-chunk, ~19.5K-char context with a large table), confirmed 8192 against that specific real case. 12 of 50 rows still fail at 8192 against a few particularly large-context questions — not chased further today; the architecturally sound fix is capping how much context reaches the judge, not raising the ceiling indefinitely.
5. **The reported total cost silently excluded the judge entirely**, caught only because the user asked directly what a run had actually cost and the honest answer was "I don't know — my own tracking never included it." DeepEval only auto-accrues cost for its own native model integrations; a custom judge's spend is otherwise invisible unless the wrapper tracks it itself. Fixed by having `ClaudeJudge` accumulate cost per call and reporting `generation_cost_usd`/`judge_cost_usd`/`total_cost_usd` separately.

None of these were hypothetical or found by reasoning about the code — each was found by actually running the thing and having it fail, crash, or (in the cost case) simply not know something it should have known.

## The real numbers
`make eval` (retrieval-only, all 50 questions, gate passes):
| Metric | Value |
|---|---|
| Recall@6 | 0.863 |
| MRR | 0.627 |

`make eval-full` (full pipeline, judge_max_tokens=8192, 38 of 50 rows judged successfully):
| Metric | Value | vs. threshold |
|---|---|---|
| Recall@6 | 0.875 | ≥0.80 ✅ |
| Citation precision | 1.000 | =1.0 ✅ |
| Refusal accuracy | 0.763 | ≥0.90 ❌ |
| Faithfulness | 0.714 | ≥0.85 ❌ |
| Correctness | 0.661 | (no threshold set) |

**Refusal accuracy and faithfulness genuinely fail their thresholds. Per CLAUDE.md, these are not lowered to pass.** Diagnosis, not guesswork:
- **Every comparison-type question (7 of 7) refused.** A single hybrid_rerank call naturally surfaces only one of two companies' relevant chunks (confirmed in `make eval`'s per-row data: these rows score exactly 0.5 recall). The model correctly judged it couldn't answer both halves and refused rather than guess — arguably the *safer* behavior for a grounded system, even though the golden set marks these `must_refuse: false`. This is a retrieval-architecture limitation (single-shot retrieval isn't suited to multi-entity comparisons), which is exactly why PLAN.md treats "comparison" as its own, harder question type.
- **A golden-set granularity gap:** `expected_sources` records only doc+section (e.g. `AAPL_2025`, Item `8`), not the specific chunk. For Apple's R&D question, Recall@6 scored 1.0 because *a* chunk from the right section was retrieved — but it was a tax footnote mentioning R&D tax *credits*, not the R&D expense line the question actually needed. The model correctly said it couldn't find the figure. Recall@6 measured "right section," not "right fact" — a real limitation of section-level ground truth, not a pipeline bug.
- **Two genuine total misses** (NVIDIA R&D, AXP forex risk) where the right section wasn't retrieved at all — a real retrieval-quality gap for these two specific questions.

## Alternative rejected
**Building a fully custom retry/backoff/rate-limit stack for the judge**, mirroring EDGAR's and Voyage's hand-rolled clients. Rejected: the per-row error-catching fix already prevents one failure from losing a run, and the actual problems here (concurrency, token limits, cost visibility) needed targeted fixes, not a bigger abstraction layer.

## Most likely failure mode
**A future large context pushing past 8192 tokens too.** The remaining 12/50 judge errors are exactly this, at smaller scale. The real fix — capping context sent to the judge rather than raising the ceiling again — is scoped but not built.

## Consequences
- **Fix-forward items for P6, not P5 blockers:** (1) cap/summarize context sent to the judge instead of raising `judge_max_tokens` further; (2) decide whether comparison-type questions need multi-query retrieval (decompose into per-entity sub-queries) or whether the golden set's `must_refuse` expectation for them should change; (3) consider chunk-level (not just section-level) `expected_sources` for tighter Recall@6 measurement.
- `evals/run_evals.py` is the shared entry point P6's experiment grid will call with different configs — nothing there needs to change structurally, just parameterizing `mode`/`settings` per run.
