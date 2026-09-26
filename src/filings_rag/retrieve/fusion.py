"""Reciprocal rank fusion: merges several ranked lists into one, favoring items
that rank well across multiple lists over an item that only ranks well in one.
"""

from collections.abc import Hashable, Sequence

from filings_rag.retrieve.models import RetrievalResult


def reciprocal_rank_fusion[T: Hashable](rankings: Sequence[Sequence[T]], k: int = 60) -> list[T]:
    """Each ranking is best-first. An item's fused score is the sum, over every
    ranking it appears in, of 1/(k + rank + 1) -- rank is 0-indexed, so 1st place
    scores highest. `k` dampens how much a single very-high rank dominates; 60 is
    the commonly cited default for this technique.
    """
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


def fuse_results(
    rankings: Sequence[Sequence[RetrievalResult]], k: int = 60
) -> list[RetrievalResult]:
    """Fuse ranked lists of RetrievalResult by id, then map back to the full
    objects (whichever ranking listed that id first)."""
    id_rankings = [[r.id for r in ranking] for ranking in rankings]
    fused_ids = reciprocal_rank_fusion(id_rankings, k=k)

    by_id: dict[str, RetrievalResult] = {}
    for ranking in rankings:
        for result in ranking:
            by_id.setdefault(result.id, result)

    return [by_id[i] for i in fused_ids]
