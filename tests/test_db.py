"""Tests for the migration runner. Needs a real Postgres (see test_db_smoke.py)."""

import psycopg
import pytest

from filings_rag.config import get_settings
from filings_rag.db import get_connection, run_migrations

pytestmark = pytest.mark.integration


@pytest.fixture
def conn() -> psycopg.Connection:
    try:
        connection = get_connection(get_settings())
    except psycopg.OperationalError as exc:
        pytest.fail(f"Cannot reach Postgres. Run `docker compose up -d db`.\n{exc}")
    # Clean slate so the test can see a real first application, not "already applied"
    # from an earlier run against the same dev database.
    connection.execute("DROP TABLE IF EXISTS chunks, schema_migrations")
    connection.commit()
    with connection:
        yield connection


def test_applies_migrations_and_tracks_them(conn: psycopg.Connection) -> None:
    applied = run_migrations(conn)
    assert applied == ["0001_create_chunks"]

    versions = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    assert versions == {"0001_create_chunks"}


def test_running_again_is_a_no_op(conn: psycopg.Connection) -> None:
    run_migrations(conn)
    assert run_migrations(conn) == []


def test_creates_expected_table_and_indexes(conn: psycopg.Connection) -> None:
    run_migrations(conn)

    columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'chunks'"
        )
    }
    assert columns == {
        "id",
        "doc_id",
        "ticker",
        "company",
        "fiscal_year",
        "item",
        "section_title",
        "page",
        "chunk_index",
        "text",
        "embedding",
        "tsv",
    }

    rows = conn.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'")
    indexes = {row[0] for row in rows}
    assert "chunks_embedding_hnsw_idx" in indexes
    assert "chunks_tsv_gin_idx" in indexes


def test_embedding_column_rejects_wrong_dimension(conn: psycopg.Connection) -> None:
    run_migrations(conn)
    with pytest.raises(psycopg.Error, match="dimension"):
        conn.execute(
            "INSERT INTO chunks "
            "(id, doc_id, ticker, company, fiscal_year, item, section_title, page, "
            " chunk_index, text, embedding) "
            "VALUES ('x', 'x', 'x', 'x', 2024, '1', 'x', 1, 0, 'hello', %s::vector)",
            (str([0.1] * 384),),  # local embedder's size, not the column's 1024
        )
