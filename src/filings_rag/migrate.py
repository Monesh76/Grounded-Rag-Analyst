"""python -m filings_rag.migrate

Applies any pending migrations to settings.database_url. Run once against a
fresh database (e.g. a newly provisioned Neon/Supabase/Cloud SQL instance)
before `make load` or serving the API -- the API itself never runs migrations
automatically on startup, so a stale schema fails loudly at query time rather
than silently changing the database on every container start.
"""

import sys

from filings_rag.config import get_settings
from filings_rag.db import get_connection, run_migrations


def main() -> int:
    settings = get_settings()
    conn = get_connection(settings)
    try:
        applied = run_migrations(conn)
    finally:
        conn.close()

    if applied:
        print(f"Applied: {', '.join(applied)}")
    else:
        print("Already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
