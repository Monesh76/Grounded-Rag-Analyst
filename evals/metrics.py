"""Deterministic golden-set metrics: Recall@k, MRR, citation precision, refusal
accuracy, latency percentiles, cost per 1K. All pure functions over plain data
(no DB, embedder or LLM), so these are unit-testable in isolation.

`doc` throughout means the golden set's TICKER_FISCALYEAR string (e.g.
"AAPL_2025", matching a filing), and `section` means the Item number (e.g.
"7"). Callers convert RetrievalResult objects to (doc, section) tuples before
calling these.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedSource:
    doc: str
    section: str


def recall_at_k(retrieved: list[tuple[str, str]], expected: list[ExpectedSource]) -> float:
    """Fraction of `expected` (doc, section) pairs found anywhere in `retrieved`
    (the caller truncates to top-k before calling). Only meaningful for
    answerable questions -- an unanswerable row has no expected sources, so
    callers should exclude those rows from this metric rather than rely on a
    default here.
    """
    if not expected:
        return 1.0
    retrieved_set = set(retrieved)
    hits = sum(1 for e in expected if (e.doc, e.section) in retrieved_set)
    return hits / len(expected)


def mean_reciprocal_rank(retrieved: list[tuple[str, str]], expected: list[ExpectedSource]) -> float:
    """Reciprocal rank (1-indexed) of the first expected source found in
    `retrieved`, in the order retrieved is given (best-first); 0.0 if none of
    them appear at all."""
    expected_set = {(e.doc, e.section) for e in expected}
    for rank, item in enumerate(retrieved, start=1):
        if item in expected_set:
            return 1.0 / rank
    return 0.0


def citation_precision(cited_ids: list[str], retrieved_ids: set[str]) -> float:
    """Fraction of `cited_ids` that were actually in the retrieved set that
    generated the answer. Measured against the model's *raw* citations, before
    citations.validate_citations() strips invalid ones -- otherwise this metric
    would tautologically always read 1.0, since the pipeline never lets an
    invalid citation reach the final Answer.
    """
    if not cited_ids:
        return 1.0
    valid = sum(1 for c in cited_ids if c in retrieved_ids)
    return valid / len(cited_ids)


def refusal_accuracy(results: list[tuple[bool, bool]]) -> float:
    """`results` is (must_refuse, actually_refused) pairs. Accuracy is the
    fraction where the pipeline's actual refuse/answer decision matched what
    the golden row expects."""
    if not results:
        return 1.0
    correct = sum(1 for expected, actual in results if expected == actual)
    return correct / len(results)


def percentile(values: list[float], p: float) -> float:
    """p in [0, 100]. Linear interpolation between closest ranks (the same
    method numpy.percentile uses by default) -- no numpy dependency needed for
    just this."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def cost_per_1k_questions(total_cost_usd: float, num_questions: int) -> float:
    if num_questions == 0:
        return 0.0
    return total_cost_usd / num_questions * 1000
