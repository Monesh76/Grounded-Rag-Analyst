"""Dense (vector similarity) search over the chunks table, via pgvector's cosine
distance operator (`<=>`), matching the HNSW index's vector_cosine_ops.
"""

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from filings_rag.embed import Embedder
from filings_rag.retrieve.common import CHUNK_TABLES, filter_conditions, row_to_result
from filings_rag.retrieve.models import Filters, RetrievalResult


def dense_search(
    conn: psycopg.Connection,
    embedder: Embedder,
    question: str,
    top_k: int,
    filters: Filters | None = None,
    table: str = "chunks",
) -> list[RetrievalResult]:
    if table not in CHUNK_TABLES:
        raise ValueError(f"Unknown chunk table {table!r}; choose one of {CHUNK_TABLES}")
    register_vector(conn)
    (raw_vector,) = embedder.embed([question])
    # pgvector's psycopg dumper is registered for its own Vector type and for
    # numpy.ndarray, not a plain Python list (which psycopg would otherwise send
    # as a double-precision array -- fine for an INSERT via an assignment cast,
    # but the `<=>` operator below needs an actual vector operand).
    query_vector = Vector(raw_vector)

    conditions, params = filter_conditions(filters)
    where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT id, doc_id, ticker, company, fiscal_year, item, section_title, page, text,
                   1 - (embedding <=> %(query_vector)s) AS score
            FROM {table}
            {where_sql}
            ORDER BY embedding <=> %(query_vector)s
            LIMIT %(top_k)s
            """,  # noqa: S608 -- table/where_sql come from our own fixed, validated allow-list
            {"query_vector": query_vector, "top_k": top_k, **params},
        )
        rows = cur.fetchall()

    return [row_to_result(row) for row in rows]
