"""Cross-encoder reranking: scores each (question, chunk) pair directly (more
accurate than comparing separately-embedded vectors, but too slow to run over
the whole table), narrowing the top-50 candidates down to a handful.
"""

from typing import Any

from filings_rag.config import Settings
from filings_rag.retrieve.models import RetrievalResult


class Reranker:
    """`model` is injectable (any object with `.predict(pairs) -> scores`) so
    tests don't need to download a real cross-encoder -- only `from_settings`
    and the default path do that."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> "Reranker":
        from sentence_transformers import CrossEncoder  # lazy: heavy import

        return cls(CrossEncoder(settings.reranker_model))

    def rerank(
        self, question: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        pairs = [(question, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [
            candidate.model_copy(update={"score": float(score)})
            for candidate, score in ranked[:top_k]
        ]
