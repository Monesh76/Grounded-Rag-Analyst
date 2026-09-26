"""Combines dense/keyword search, fusion and reranking behind one configurable
entry point, so retrieval quality can be compared across modes (PLAN.md's
experiment grid) without changing calling code.
"""

import psycopg

from filings_rag.config import Settings
from filings_rag.embed import Embedder
from filings_rag.retrieve.dense import dense_search
from filings_rag.retrieve.filters import extract_filters
from filings_rag.retrieve.fusion import fuse_results
from filings_rag.retrieve.keyword import keyword_search
from filings_rag.retrieve.models import RetrievalResult
from filings_rag.retrieve.rerank import Reranker

MODES = ("dense", "keyword", "hybrid", "hybrid_rerank")


def search(
    conn: psycopg.Connection,
    embedder: Embedder,
    reranker: Reranker | None,
    question: str,
    mode: str,
    settings: Settings,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; choose one of {MODES}")

    top_k = top_k or settings.retrieval_rerank_top_k
    filters = extract_filters(question)

    if mode == "dense":
        return dense_search(conn, embedder, question, settings.retrieval_dense_top_k, filters)[
            :top_k
        ]

    if mode == "keyword":
        return keyword_search(conn, question, settings.retrieval_keyword_top_k, filters)[:top_k]

    dense_results = dense_search(conn, embedder, question, settings.retrieval_dense_top_k, filters)
    keyword_results = keyword_search(conn, question, settings.retrieval_keyword_top_k, filters)
    fused = fuse_results([dense_results, keyword_results], k=settings.retrieval_rrf_k)

    if mode == "hybrid":
        return fused[:top_k]

    if reranker is None:
        raise ValueError("mode 'hybrid_rerank' requires a reranker")
    # Rerank over the full fused pool (up to top-50), not just the final top_k --
    # otherwise reranking could only ever reorder within a list already cut down
    # by fusion, defeating the point of scoring each candidate against the question.
    return reranker.rerank(question, fused[: settings.retrieval_dense_top_k], top_k=top_k)
