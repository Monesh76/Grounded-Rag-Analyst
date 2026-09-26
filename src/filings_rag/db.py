"""Postgres connection and a small hand-rolled migration runner.

No migration framework (e.g. Alembic) for one table: applying numbered .sql files
in order and tracking which ran in a `schema_migrations` table is a dozen lines
and is easy to read end to end.
"""

import re
from pathlib import Path
from urllib.parse import urlsplit

import psycopg

from filings_rag.config import Settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"


def get_connection(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.database_url)


def get_test_connection(settings: Settings) -> psycopg.Connection:
    """Connection to the separate test database (see Settings.test_database_url),
    creating it first if it doesn't exist yet."""
    _ensure_database_exists(settings.test_database_url)
    return psycopg.connect(settings.test_database_url)


def _ensure_database_exists(database_url: str) -> None:
    db_name = urlsplit(database_url).path.lstrip("/")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", db_name):
        raise ValueError(f"Unexpected database name in test_database_url: {db_name!r}")

    admin_url = database_url.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin_url, autocommit=True) as admin_conn:
        exists = admin_conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
        ).fetchone()
        if not exists:
            # CREATE DATABASE can't be parameterized or run inside a transaction;
            # db_name is validated above, so this is safe to interpolate.
            admin_conn.execute(f'CREATE DATABASE "{db_name}"')  # noqa: S608


def run_migrations(conn: psycopg.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply any .sql files in `migrations_dir` not yet recorded as applied.
    Returns the versions newly applied (empty if the schema was already current).

    Each migration is executed and committed on its own -- not wrapped with its
    schema_migrations bookkeeping insert in one transaction -- because every
    statement in a migration file uses IF NOT EXISTS, making the file safe to
    re-run. That idempotency is what makes it fine to not need atomicity here: if
    a crash lands between "migration applied" and "recorded as applied", the next
    run just re-applies it (a no-op) and then records it correctly.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now()"
        ")"
    )
    conn.commit()
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}

    newly_applied = []
    for path in sorted(migrations_dir.glob("*.sql")):
        version = path.stem
        if version in applied:
            continue
        conn.execute(path.read_text())
        conn.commit()
        conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
        conn.commit()
        newly_applied.append(version)
    return newly_applied
