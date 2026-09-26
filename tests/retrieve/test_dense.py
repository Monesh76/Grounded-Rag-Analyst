"""Integration tests for dense_search. Uses hand-picked unit vectors so cosine
distance is exact and predictable, and a fake Embedder so no real embedding
call is needed for the query.
"""

import psycopg
import pytest

from filings_rag.retrieve.dense import dense_search
from filings_rag.retrieve.models import Filters
from tests.retrieve.conftest import insert_chunk, unit_vector

pytestmark = pytest.mark.integration


class FixedEmbedder:
    """Always returns the same vector, regardless of the query text -- lets the
    test control exactly what dense_search compares against."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    dimensions = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


def test_orders_by_cosine_distance_closest_first(conn: psycopg.Connection) -> None:
    query_vector = unit_vector(0)
    insert_chunk(conn, "identical", "matches exactly", unit_vector(0))  # distance 0
    insert_chunk(conn, "orthogonal", "unrelated", unit_vector(1))  # distance 1
    insert_chunk(conn, "opposite", "opposite direction", [-x for x in unit_vector(0)])  # distance 2

    results = dense_search(conn, FixedEmbedder(query_vector), "query", top_k=10)
    ids = [r.id for r in results if r.id in ("identical", "orthogonal", "opposite")]

    assert ids == ["identical", "orthogonal", "opposite"]
    assert results[0].score == pytest.approx(1.0)  # 1 - distance(0) = 1


def test_respects_top_k(conn: psycopg.Connection) -> None:
    for i in range(5):
        insert_chunk(conn, f"c{i}", f"chunk {i}", unit_vector(i % 1024))

    results = dense_search(conn, FixedEmbedder(unit_vector(0)), "query", top_k=2)
    assert len(results) == 2


def test_applies_ticker_filter(conn: psycopg.Connection) -> None:
    insert_chunk(conn, "match", "text", unit_vector(0), ticker="TEST")
    insert_chunk(conn, "other", "text", unit_vector(0), ticker="OTHER")

    results = dense_search(
        conn, FixedEmbedder(unit_vector(0)), "query", top_k=10, filters=Filters(ticker="TEST")
    )

    assert [r.id for r in results] == ["match"]


def test_applies_fiscal_year_filter(conn: psycopg.Connection) -> None:
    insert_chunk(conn, "y2023", "text", unit_vector(0), fiscal_year=2023)
    insert_chunk(conn, "y2024", "text", unit_vector(0), fiscal_year=2024)

    results = dense_search(
        conn, FixedEmbedder(unit_vector(0)), "query", top_k=10, filters=Filters(fiscal_year=2024)
    )

    assert [r.id for r in results] == ["y2024"]
