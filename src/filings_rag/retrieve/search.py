"""Combines dense/keyword search, fusion and reranking behind one configurable
entry point, so retrieval quality can be compared across modes (PLAN.md's
experiment grid) without changing calling code.
"""

import psycopg

from filings_rag import tracing
from filings_rag.config import Settings
from filings_rag.embed import Embedder
from filings_rag.retrieve.dense import dense_search
from filings_rag.retrieve.filters import extract_filters
from filings_rag.retrieve.fusion import fuse_results
from filings_rag.retrieve.keyword import keyword_search
from filings_rag.retrieve.models import Filters, RetrievalResult
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
    table: str = "chunks",
    filters: Filters | None = None,
) -> list[RetrievalResult]:
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; choose one of {MODES}")

    top_k = top_k or settings.retrieval_rerank_top_k
    # An explicit filter (e.g. the UI's company/year dropdowns) always wins
    # over guessing one from the question text.
    filters = filters if filters is not None else extract_filters(question)

    if mode == "dense":
        return _dense(conn, embedder, question, filters, settings, table)[:top_k]

    if mode == "keyword":
        return _keyword(conn, question, filters, settings, table)[:top_k]

    dense_results = _dense(conn, embedder, question, filters, settings, table)
    keyword_results = _keyword(conn, question, filters, settings, table)
    fused = fuse_results([dense_results, keyword_results], k=settings.retrieval_rrf_k)

    if mode == "hybrid":
        return fused[:top_k]

    if reranker is None:
        raise ValueError("mode 'hybrid_rerank' requires a reranker")
    # Rerank over the full fused pool (up to top-50), not just the final top_k --
    # otherwise reranking could only ever reorder within a list already cut down
    # by fusion, defeating the point of scoring each candidate against the question.
    with tracing.span(settings, "rerank", as_type="span", input={"question": question}) as obs:
        reranked = reranker.rerank(question, fused[: settings.retrieval_dense_top_k], top_k=top_k)
        tracing.update(obs, output={"count": len(reranked)})
    return reranked


def _dense(
    conn: psycopg.Connection,
    embedder: Embedder,
    question: str,
    filters: Filters,
    settings: Settings,
    table: str,
) -> list[RetrievalResult]:
    with tracing.span(
        settings, "dense-retrieval", as_type="retriever", input={"question": question}
    ) as obs:
        results = dense_search(
            conn, embedder, question, settings.retrieval_dense_top_k, filters, table
        )
        tracing.update(obs, output={"count": len(results)})
    return results


def _keyword(
    conn: psycopg.Connection, question: str, filters: Filters, settings: Settings, table: str
) -> list[RetrievalResult]:
    with tracing.span(
        settings, "keyword-retrieval", as_type="retriever", input={"question": question}
    ) as obs:
        results = keyword_search(conn, question, settings.retrieval_keyword_top_k, filters, table)
        tracing.update(obs, output={"count": len(results)})
    return results
