"""Integration tests for keyword_search against real Postgres full-text search."""

import psycopg
import pytest

from filings_rag.retrieve.filters import extract_filters
from filings_rag.retrieve.keyword import keyword_search
from tests.retrieve.conftest import insert_chunk, unit_vector

pytestmark = pytest.mark.integration


def test_matches_on_keyword_content(conn: psycopg.Connection) -> None:
    insert_chunk(conn, "supply", "supply chain disruption is a key risk", unit_vector(0))
    insert_chunk(conn, "unrelated", "net revenue increased year over year", unit_vector(1))

    results = keyword_search(conn, "supply chain risk", top_k=10)

    assert [r.id for r in results] == ["supply"]


def test_ranks_more_relevant_text_higher(conn: psycopg.Connection) -> None:
    insert_chunk(conn, "strong", "revenue revenue revenue growth", unit_vector(0))
    insert_chunk(conn, "weak", "a brief mention of revenue somewhere", unit_vector(1))

    results = keyword_search(conn, "revenue", top_k=10)

    assert results[0].id == "strong"


def test_no_match_returns_empty(conn: psycopg.Connection) -> None:
    insert_chunk(conn, "a", "supply chain risk", unit_vector(0))
    results = keyword_search(conn, "nonexistent gibberish zzqx", top_k=10)
    assert results == []


def test_applies_metadata_filters(conn: psycopg.Connection) -> None:
    insert_chunk(conn, "match", "net revenue details", unit_vector(0), fiscal_year=2024)
    insert_chunk(conn, "other_year", "net revenue details", unit_vector(1), fiscal_year=2023)

    filters = extract_filters("What was net revenue in 2024?")
    results = keyword_search(conn, "net revenue", top_k=10, filters=filters)

    assert [r.id for r in results] == ["match"]
