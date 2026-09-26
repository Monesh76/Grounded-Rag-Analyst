# 005 — Generation, citations and API (P4)

**Status:** accepted · 2026-09-25

## Context
P3 leaves a working retrieval path. P4 turns retrieved chunks into an actual answer: a system prompt that cites sources, an evidence gate that refuses rather than guesses, a citation validator that can't be fooled by a fabricated id, and a FastAPI endpoint — verified against real curl requests, not just mocked tests.

## Decisions
- **Three LLM backends behind one protocol:** Claude (production default), OpenAI (PLAN.md's named second provider), and OpenRouter (an OpenAI-compatible proxy, reusing the same client code with a different `base_url`/model — no third implementation needed). OpenRouter defaults to `anthropic/claude-haiku-4.5` and is the *default* `llm_provider`, per the user's explicit workflow: cheap during dev/testing, `claude`/`openai` set explicitly for final checks and production.
- **Evidence gate short-circuits before any LLM call** when the top rerank score is below `evidence_gate_threshold` — deterministic, zero cost, and it's what makes the off-topic-question test show `tokens: 0`.
- **Citation validation strips, not just flags, an unknown id.** A fabricated `[c:<id>]` never reaches the user looking like a real source.
- **`grounded` requires all three:** the LLM's answer isn't the refusal, it has zero invalid citations, and it has at least one valid citation. A model that answers without citing anything (despite being told to cite every claim) is treated as ungrounded, not "probably fine."
- **Streaming with the citation guarantee intact:** `pipeline.ask()` runs the LLM call and validates the full response *before* the API ever sees it; `stream=true` sends that already-validated text to the client in pieces, then one final JSON line with the structured metadata. A true token-by-token pass-through from the LLM was rejected on purpose — it could show unvalidated (possibly fabricated-citation) text before validation could catch it.

## Two real bugs, both only visible by calling the actual model
1. **The prompt's "reply with exactly this sentence and nothing else" instruction wasn't followed.** A real call, judging the evidence insufficient, produced the refusal sentence followed by an explanatory paragraph. CLAUDE.md's contract is an *exact* string, so this can't be left to the prompt. Fixed by truncating any response that *starts with* the refusal text to exactly that string in code, discarding whatever followed, and still reporting the real tokens/cost from that call.
2. **The installed Anthropic SDK (Claude 5 family) has no `temperature` parameter on `Messages.create`** — replaced by `output_config.effort`, a different axis (reasoning effort, not sampling randomness). Found by inspecting the live SDK signature after a `TypeError`, not by assuming the API shape from prior knowledge. `ClaudeProvider` no longer passes it; the OpenAI-compatible providers still do.

Also hit an account-level issue with the same shape as Voyage's rate-limit surprise in P2: the initial Anthropic API key wasn't scoped to a workspace, which the API now requires. Not fixable in code — resolved by the user swapping in a workspace-scoped key.

## The actual acceptance check
Ran `make serve` and used real `curl` requests against the live API (not TestClient, not mocks):
- **A real question** ("What was Visa's net revenue in fiscal 2025?") → grounded, cited, correct answer with real token/cost figures.
- **An off-topic question** ("chocolate chip cookie recipe") → the exact refusal string, `tokens: 0` (evidence gate fired before any LLM call).

Both match PLAN.md's P4 "Done when."

## A performance observation worth carrying forward
Real per-request latency was ~22-25 seconds. `pipeline.ask()` currently rebuilds the DB connection, embedder, and reranker from scratch on every call — the reranker model reload is the likely dominant cost. Not fixed here (out of scope for P4's correctness goals), but a real production concern: the standard fix is request-scoped dependency injection of long-lived singletons (FastAPI's `lifespan`/app state), reused across requests instead of rebuilt per request.

## Alternative rejected
**A true token-by-token streaming pass-through from the LLM.** Would satisfy "streams tokens" more literally, but breaks the citation guarantee: the client could see a fabricated citation mid-stream before validation ever ran. Buffering server-side and streaming the validated result is the correct tradeoff for a system whose whole premise is "never show an unverifiable claim."

## Most likely failure mode
**A model phrasing a refusal differently than expected.** The truncation fix only catches a response that *starts with* the exact configured refusal text; a model that refuses with different wording, different casing, or a refusal buried mid-response would fall through to ordinary citation validation instead — which should still mark it ungrounded (no valid citations), but wouldn't produce the clean single-sentence refusal CLAUDE.md asks for. The evidence gate is the primary defense against this; this fallback is a secondary, string-matching-based safety net with real limits.

## Consequences
- P5's eval harness calls `pipeline.ask()` directly, the same entry point the API uses — no separate code path to keep in sync.
- The `evidence_gate_threshold` default (0.1) is still a provisional placeholder, flagged in config.py; P5's golden set is what it should actually be tuned against.
