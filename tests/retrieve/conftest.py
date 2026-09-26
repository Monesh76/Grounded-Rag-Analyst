"""Shared fixtures for dense/keyword search integration tests."""

import psycopg
import pytest
from pgvector.psycopg import register_vector

from filings_rag.config import get_settings
from filings_rag.db import get_connection, run_migrations

DIMENSIONS = 1024


def unit_vector(index: int) -> list[float]:
    """A vector with a 1 at `index` and 0 elsewhere -- lets tests reason about
    cosine distance exactly (e.g. two identical unit vectors -> distance 0)."""
    v = [0.0] * DIMENSIONS
    v[index] = 1.0
    return v


@pytest.fixture
def conn() -> psycopg.Connection:
    try:
        connection = get_connection(get_settings())
    except psycopg.OperationalError as exc:
        pytest.fail(f"Cannot reach Postgres. Run `docker compose up -d db`.\n{exc}")
    run_migrations(connection)
    register_vector(connection)
    # Explicit commit()s throughout this fixture and insert_chunk, not
    # `with connection:`/`conn.transaction()`: a bare execute() (like the SELECTs
    # inside dense_search/keyword_search) left open before one of those context
    # managers silently breaks its commit -- see docs/decisions/003.
    connection.execute("DELETE FROM chunks WHERE ticker IN ('TEST', 'OTHER')")
    connection.commit()
    yield connection
    connection.execute("DELETE FROM chunks WHERE ticker IN ('TEST', 'OTHER')")
    connection.commit()
    connection.close()


def insert_chunk(
    conn: psycopg.Connection,
    id: str,
    text: str,
    embedding: list[float],
    ticker: str = "TEST",
    fiscal_year: int = 2024,
    item: str = "1",
) -> None:
    conn.execute(
        """
        INSERT INTO chunks (id, doc_id, ticker, company, fiscal_year, item,
                             section_title, page, chunk_index, text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 1, 0, %s, %s)
        """,
        (id, "doc", ticker, "Test Co", fiscal_year, item, "Section", text, embedding),
    )
    conn.commit()
