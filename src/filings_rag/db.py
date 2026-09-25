"""Postgres connection and a small hand-rolled migration runner.

No migration framework (e.g. Alembic) for one table: applying numbered .sql files
in order and tracking which ran in a `schema_migrations` table is a dozen lines
and is easy to read end to end.
"""

from pathlib import Path

import psycopg

from filings_rag.config import Settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"


def get_connection(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.database_url)


def run_migrations(conn: psycopg.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply any .sql files in `migrations_dir` not yet recorded as applied.
    Returns the versions newly applied (empty if the schema was already current).
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now()"
        ")"
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}

    newly_applied = []
    for path in sorted(migrations_dir.glob("*.sql")):
        version = path.stem
        if version in applied:
            continue
        with conn.transaction():
            conn.execute(path.read_text())
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
        newly_applied.append(version)
    return newly_applied
