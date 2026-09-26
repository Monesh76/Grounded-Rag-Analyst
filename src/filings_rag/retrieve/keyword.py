"""Keyword (full-text) search over the chunks table, via the generated `tsv`
column and `websearch_to_tsquery` -- the same query syntax a web search box
accepts ("quotes", -exclude, OR), which is friendlier for a question than
`plainto_tsquery`.
"""

import psycopg
from psycopg.rows import dict_row

from filings_rag.retrieve.common import filter_conditions, row_to_result
from filings_rag.retrieve.models import Filters, RetrievalResult


def keyword_search(
    conn: psycopg.Connection,
    question: str,
    top_k: int,
    filters: Filters | None = None,
) -> list[RetrievalResult]:
    conditions, params = filter_conditions(filters)
    conditions.insert(0, "tsv @@ websearch_to_tsquery('english', %(question)s)")
    params["question"] = question

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT id, doc_id, ticker, company, fiscal_year, item, section_title, page, text,
                   ts_rank_cd(tsv, websearch_to_tsquery('english', %(question)s)) AS score
            FROM chunks
            WHERE {" AND ".join(conditions)}
            ORDER BY score DESC
            LIMIT %(top_k)s
            """,  # noqa: S608 -- conditions are our own fixed strings, question is parameterized
            {"top_k": top_k, **params},
        )
        rows = cur.fetchall()

    return [row_to_result(row) for row in rows]
