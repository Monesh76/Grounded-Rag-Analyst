"""python -m filings_rag.seed_eval_fixture

Applies migrations then loads tests/fixtures/eval_seed.sql -- a small set of
real, already-embedded chunks -- into `settings.database_url`. This is what CI
runs before `make eval-ci`, so retrieval has real data to search without
running the real ingest/embed/load pipeline (money, SEC/Voyage keys) on every PR.
"""

import sys
from pathlib import Path

from filings_rag.config import get_settings
from filings_rag.db import get_connection, run_migrations

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eval_seed.sql"


def main() -> int:
    settings = get_settings()
    conn = get_connection(settings)
    try:
        run_migrations(conn)
        conn.execute(FIXTURE_PATH.read_text())
        conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
