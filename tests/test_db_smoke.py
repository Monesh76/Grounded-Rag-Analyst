"""Smoke test: Postgres is reachable and the pgvector extension works."""

from collections.abc import Iterator

import psycopg
import pytest

from filings_rag.config import get_settings

pytestmark = pytest.mark.integration


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    url = get_settings().database_url
    try:
        connection = psycopg.connect(url, connect_timeout=5, autocommit=True)
    except psycopg.OperationalError as exc:
        # Fail (not skip) so a missing DB in CI can't hide behind a green run.
        pytest.fail(f"Cannot reach Postgres at {url}. Run `docker compose up -d db`.\n{exc}")
    with connection:
        yield connection


def test_vector_extension_available(conn: psycopg.Connection) -> None:
    # The image ships pgvector but doesn't enable it; IF NOT EXISTS makes this idempotent.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert row is not None, "vector extension is not installed"

    # L2 distance between [1,2,3] and [1,2,4] is exactly 1.
    (distance,) = conn.execute("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector").fetchone()
    assert distance == pytest.approx(1.0)
