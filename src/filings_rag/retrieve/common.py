"""Shared bits between dense.py and keyword.py: building a WHERE clause from
Filters, and mapping a result row back to a RetrievalResult.
"""

from typing import Any

from filings_rag.retrieve.models import Filters, RetrievalResult

# The only tables dense/keyword search are ever pointed at -- "chunks"
# (production, section-aware, Voyage 1024-dim), "chunks_fixed512" (P6
# experiment A's fixed-chunking baseline, same embedder/dims), and
# "chunks_local" (CI's fixture, local sentence-transformers 384-dim, so
# eval-ci needs zero API calls -- not even to embed the incoming question).
# An allow-list, not just internal-only-string trust, since the table name
# gets interpolated directly into SQL (identifiers can't be parameterized the
# way values can).
CHUNK_TABLES = ("chunks", "chunks_fixed512", "chunks_local")


def filter_conditions(filters: Filters | None) -> tuple[list[str], dict[str, Any]]:
    """SQL condition strings (no WHERE/AND) and their params, for whichever
    filters are set. Empty when `filters` is None or nothing was extracted."""
    conditions: list[str] = []
    params: dict[str, Any] = {}
    if filters:
        if filters.ticker:
            conditions.append("ticker = %(ticker)s")
            params["ticker"] = filters.ticker
        if filters.fiscal_year:
            conditions.append("fiscal_year = %(fiscal_year)s")
            params["fiscal_year"] = filters.fiscal_year
    return conditions, params


def row_to_result(row: dict[str, Any]) -> RetrievalResult:
    return RetrievalResult(
        id=row["id"],
        doc_id=row["doc_id"],
        ticker=row["ticker"],
        company=row["company"],
        fiscal_year=row["fiscal_year"],
        item=row["item"],
        section_title=row["section_title"],
        page=row["page"],
        text=row["text"],
        score=row["score"],
    )
