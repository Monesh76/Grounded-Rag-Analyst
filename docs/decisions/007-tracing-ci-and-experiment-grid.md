# 007 — Tracing, CI gates and the experiment grid (P6)

**Status:** accepted · 2026-09-26

## Context
P6 adds observability (Langfuse tracing), a CI gate that runs on every PR for free, a paid full-eval workflow gated behind a label/schedule, and PLAN.md's A-E experiment grid comparing chunking and retrieval strategies. Like P5, most of what's genuinely worth recording here is what running things for real surfaced that reasoning about the code wouldn't have.

## Decisions
- **Manual Langfuse spans (`start_as_current_observation`), not the `@observe` decorator.** This project builds its own `Langfuse` client from `settings` (config.py's "no keys read outside config.py" rule), not the env-var-based global client `@observe` relies on — explicit spans are the more predictable integration given that.
- **Tracing is a true no-op without `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`** — `tracing.span()` returns `None` and every caller goes through `tracing.update()`, which no-ops on `None`. `make test` and CI's default job never need Langfuse credentials.
- **CI's free job (`make eval-ci`) runs against a small committed fixture** (`tests/fixtures/eval_seed.sql`: 89 real chunk rows, real Voyage embeddings already paid for in P2/P5, dumped straight from the loaded dev DB) rather than the real ingest/embed/load pipeline — zero API calls in CI, ever, on every PR.
- **`evals-full.yml` (the paid path) is gated behind the `run-full-evals` label and a nightly cron**, never a push. It needs `EVAL_DATABASE_URL` plus provider keys as repository secrets, which were not added — provisioning a persistent, GitHub-Actions-reachable Postgres is infrastructure work outside this repo, flagged rather than assumed done.
- **Fixed-512 chunking (experiment A) loads into a separate `chunks_fixed512` table**, not a column on `chunks` — a chunk id collision between two different chunking strategies producing the same `TICKER_YEAR_ITEM_INDEX` id would silently overwrite production rows otherwise.
- **The comparison table reports generation-only cost, not total cost, and adds a "Judged" column.** Total cost includes judge calls for judged runs and doesn't for unjudged ones; comparing it directly across a mix (which the grid ended up being) would make hybrid_rerank look 15-100x cheaper than dense retrieval for reasons having nothing to do with retrieval quality.

## The bugs and real findings, in the order they were found
1. **`langfuse_base_url` field name didn't match the env var actually set.** `config.py` originally named the field `langfuse_host`, which pydantic-settings maps to `LANGFUSE_HOST` — but `.env` (and Langfuse's own SDK convention) sets `LANGFUSE_BASE_URL`. The mismatch silently fell back to the wrong region's default host; every trace export failed with `401 Unauthorized` instead of a clear "not configured" error. Found by actually running a real retrieval call end to end and watching the export fail, not by inspection — fixed by renaming the field to match the real env var.
2. **A ~13x underestimate of the paid comparison grid's real cost, on me.** I estimated $1-2 for all five configs based on decision record 006's "$0.20 total" figure from the last P5 run. That figure was itself wrong: it predated the judge-cost-tracking fix from earlier this session, so it silently excluded every judge call and was really just generation cost mislabeled as total. The real driver: DeepEval's `FaithfulnessMetric` makes several LLM calls per question internally (extract claims, extract truths from the retrieved context, generate verdicts), plus one more for `GEval` correctness — all at Claude Sonnet 5's pricing. Judge cost ran 10-20x generation cost in every judged run. Real spend on A-D before this was caught: **$15.12** (A: $3.70, B: $4.82, C: $4.72, D: $1.88 before the next bug stopped it).
3. **The Anthropic account ran out of credit mid-run**, 24 of 50 rows into config D's judge pass (`Error code: 400 ... Your credit balance is too low`). Caught per-row by P5's existing error-catching fix (a real payoff of that earlier fix: this failure did not lose the other 47 rows), but it made D's numbers at that point unreliable — computed over a biased ~24-row subset, not the intended 50. Run E never started; the whole grid was stopped once the true cost was understood, rather than continuing to spend while re-deriving the estimate.
4. **A user-facing tradeoff, not a bug: D and E were finished with `--no-judge`.** Given the $15.12 already spent and no appetite for another ~$7-9 to judge D+E properly, the user chose to complete the grid's retrieval-comparison metrics (Recall@6, MRR, citation precision, refusal accuracy, latency, generation cost — the metrics the grid is actually about) for free, and accept "n/a" for faithfulness/correctness on those two rows. A/B/C's faithfulness/correctness are real judged numbers from before the credit exhaustion; D/E's are not measured.
5. **A transient ~40x latency inflation, confirmed as machine contention, not a code bug.** D and E's first `--no-judge` run showed p95 latency of ~190 seconds, vs. 3-9 seconds for every other run including the earlier judged runs of the same configs. Rather than report that uncritically, re-ran D alone immediately after: p95 dropped to 9.2s, back in the normal range. The likely cause: this session had run many concurrent CPU/network-heavy things back to back (the fixed-chunking re-embed, the failed 5-config judged run, this machine's normal load from VS Code and other apps — `uptime` showed a load average of 2.7 at the time) contending with `run_full`'s shared `_MODEL_LOCK`, which serializes every row's embed+rerank step within a run. Re-ran E the same way once D confirmed the pattern. Both configs' final numbers in `comparison.md` are from the clean reruns.

## The real numbers
`results/comparison.md` (generation cost, not total — see decisions above):

| Run | Chunking | Retrieval | Model | Judged | Recall@6 | MRR | Faithfulness | Correctness | Refusal acc. | p95 latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | fixed-512 | dense | claude-haiku-4.5 | yes | 0.838 | 0.632 | 0.800 | 0.733 | 0.840 | 4096 |
| B | section-aware | dense | claude-haiku-4.5 | yes | 0.787 | 0.602 | 0.850 | 0.760 | 0.860 | 5509 |
| C | section-aware | hybrid | claude-haiku-4.5 | yes | 0.787 | 0.572 | 0.850 | 0.745 | 0.860 | 3426 |
| D | section-aware | hybrid+rerank | claude-haiku-4.5 | no | 0.850 | 0.619 | n/a | n/a | 0.760 | 9211 |
| E | section-aware | hybrid+rerank | gpt-4o-mini | no | 0.850 | 0.619 | n/a | n/a | 0.780 | 16565 |

Real patterns visible in this data, not just numbers for their own sake:
- **Section-aware chunking (B) beats fixed-512 (A) on refusal accuracy at equal recall** (0.860 vs. 0.840, both at 0.787/0.838 recall) — the one clean signal the grid was built to produce, and it points the direction PLAN.md's chunking choice already went.
- **Reranking (D) gives the best Recall@6/MRR of any config** (0.850/0.619), at the cost of the *worst* refusal accuracy (0.760) — consistent with decision record 006's finding that hybrid_rerank's comparison-question refusals, not a fusion or chunking defect, are the main drag on that metric.
- **D vs. E confirms retrieval metrics are model-independent, as expected**: identical Recall@6/MRR (both 0.850/0.619) since only the generation model differs between them — a sanity check that the harness is measuring what it claims to.

## Alternative rejected
**Continuing to spend on a fully judged D+E** rather than accepting "n/a" for two rows. Rejected by the user's explicit choice once the real per-run judge cost was known: the grid's actual purpose (comparing retrieval/chunking strategies) is already answered by Recall@6/MRR, which don't need a judge at all.

## Most likely failure mode
**A future paid run silently costing far more than expected, the same way this one did**, if `run_full`'s judge cost isn't estimated *before* running rather than after. The real fix, not yet built: a `--estimate` flag or a documented per-question judge-cost figure (now known: roughly $0.07-0.10/question for Claude Sonnet 5 via DeepEval's FaithfulnessMetric + GEval) so a future cost estimate is derived from this session's real numbers, not from a stale or wrong figure the way this one was.

## Consequences
- **Fix-forward items for P7, not P6 blockers:** (1) cap/summarize context sent to the judge (already flagged in 006, now doubly motivated — it would also lower judge cost, not just fix truncation); (2) provision a persistent, GitHub-Actions-reachable Postgres and add the repo secrets `evals-full.yml` needs, if nightly full evals are wanted; (3) judge D and E properly if/when there's appetite for the ~$7-9 cost, to get real faithfulness/correctness for the full grid.
- `evals/run_comparison.py`'s `--only` flag (added mid-P6, after the credit exhaustion) is now the standing way to re-run part of the grid without re-paying for parts that already have real numbers — useful beyond this one incident.
