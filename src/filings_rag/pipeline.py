"""ask(question, settings) -> Answer: retrieve, gate on evidence quality,
generate with citations, validate, return. The one entry point the API (and
the eval harness) calls -- nothing downstream duplicates this logic.

ask() is a thin wrapper over ask_detailed(), which also returns what was
retrieved and the LLM's raw (pre-validation) text -- the API doesn't need
those, but the eval harness does, to measure Recall@k/MRR against retrieval
and citation precision against the model's *unvalidated* output.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from filings_rag import tracing
from filings_rag.config import Settings
from filings_rag.db import get_connection
from filings_rag.embed import get_embedder
from filings_rag.generate.citations import validate_citations
from filings_rag.generate.llm import LLMProvider, get_llm_provider
from filings_rag.generate.models import Answer
from filings_rag.generate.prompts import build_system_prompt, build_user_message
from filings_rag.retrieve.models import Filters, RetrievalResult
from filings_rag.retrieve.rerank import Reranker
from filings_rag.retrieve.search import search


@dataclass
class AskResult:
    answer: Answer
    retrieved: list[RetrievalResult]  # empty if the evidence gate refused first
    raw_answer_text: str | None  # the LLM's own text before citation validation;
    # None if the LLM was never called (evidence gate refused)


def ask(
    question: str,
    settings: Settings,
    llm: LLMProvider | None = None,
    retrieve: Callable[[str], list[RetrievalResult]] | None = None,
    filters: Filters | None = None,
) -> Answer:
    return ask_detailed(question, settings, llm=llm, retrieve=retrieve, filters=filters).answer


def ask_detailed(
    question: str,
    settings: Settings,
    llm: LLMProvider | None = None,
    retrieve: Callable[[str], list[RetrievalResult]] | None = None,
    filters: Filters | None = None,
) -> AskResult:
    """`llm` and `retrieve` are injectable for testing (a fake LLM, a fake
    retrieval function returning canned chunks) -- each defaults to the real
    thing built from `settings` when not given. `filters` (e.g. the UI's
    company/year dropdowns) is ignored when `retrieve` is injected -- the
    caller's own retrieval function is responsible for filtering then."""
    start = time.monotonic()
    llm = llm or get_llm_provider(settings)

    owns_conn = retrieve is None
    conn = None
    if retrieve is None:
        conn = get_connection(settings)
        embedder = get_embedder(settings)
        reranker = Reranker.from_settings(settings)
        retrieve = lambda q: search(  # noqa: E731
            conn, embedder, reranker, q, "hybrid_rerank", settings, filters=filters
        )

    with tracing.span(settings, "ask", as_type="span", input={"question": question}) as root:
        try:
            results = retrieve(question)

            # Evidence gate: refuse without ever calling the LLM if retrieval itself
            # is too weak to plausibly answer -- deterministic, and saves the cost of
            # a call that would likely need to refuse anyway.
            if not results or results[0].score < settings.evidence_gate_threshold:
                answer = _refusal(settings, start)
                tracing.update(root, output=answer.answer, metadata={"refused": True})
                return AskResult(answer=answer, retrieved=[], raw_answer_text=None)

            system = build_system_prompt(settings.refusal_text)
            user = build_user_message(question, results)
            with tracing.span(
                settings, "generate", as_type="generation", input={"question": question}
            ) as gen:
                response = llm.complete(system, user)
                tracing.update(
                    gen,
                    output=response.text,
                    model=llm.model,
                    usage_details={
                        "input": response.input_tokens,
                        "output": response.output_tokens,
                    },
                    metadata={"cost_usd": response.cost_usd},
                )

            # The prompt asks for exactly the refusal sentence "and nothing else", but
            # models don't reliably follow that -- observed in practice: a model
            # judging the evidence insufficient still padded the refusal with an
            # explanatory paragraph. CLAUDE.md requires the exact string, so this is
            # enforced here rather than trusted to the prompt: a response that leads
            # with the refusal is truncated to exactly that string, discarding
            # whatever the model added after it. A model phrasing the refusal
            # differently (no trailing period, different casing) won't be caught by
            # this check and falls through to citation validation instead, where the
            # lack of any citations should still mark it ungrounded.
            text = response.text.strip()
            if text.startswith(settings.refusal_text):
                refusal = _refusal(
                    settings,
                    start,
                    tokens=response.input_tokens + response.output_tokens,
                    cost_usd=response.cost_usd,
                )
                tracing.update(root, output=refusal.answer, metadata={"refused": True})
                return AskResult(answer=refusal, retrieved=results, raw_answer_text=response.text)

            citations = validate_citations(response.text, results)
            # Conservative: a model that cited even one fabricated id is treated as
            # ungrounded, not "grounded but slightly wrong" -- silently correcting a
            # hallucinated citation isn't the same as trusting the rest of the answer.
            # An answer with no citations at all despite being asked to cite every
            # claim is also not grounded.
            grounded = not citations.has_invalid_citations and bool(citations.sources)

            answer = Answer(
                answer=citations.text,
                sources=citations.sources,
                grounded=grounded,
                latency_ms=(time.monotonic() - start) * 1000,
                tokens=response.input_tokens + response.output_tokens,
                cost_usd=response.cost_usd,
            )
            tracing.update(
                root, output=answer.answer, metadata={"grounded": grounded, "refused": False}
            )
            return AskResult(answer=answer, retrieved=results, raw_answer_text=response.text)
        finally:
            if owns_conn and conn is not None:
                conn.close()


def _refusal(settings: Settings, start: float, tokens: int = 0, cost_usd: float = 0.0) -> Answer:
    return Answer(
        answer=settings.refusal_text,
        sources=[],
        grounded=False,
        latency_ms=(time.monotonic() - start) * 1000,
        tokens=tokens,
        cost_usd=cost_usd,
    )
