"""python -m filings_rag.seed_eval_fixture

Applies migrations then loads both eval fixtures -- the same 89 real chunks,
embedded two ways -- into `settings.database_url`:

- tests/fixtures/eval_seed.sql -> chunks (Voyage, 1024-dim)
- tests/fixtures/eval_seed_local.sql -> chunks_local (local sentence-
  transformers, 384-dim)

CI's `make eval-ci` uses chunks_local specifically: dense retrieval embeds the
incoming *question* at query time on every call, not just the corpus once, so
even a fully pre-embedded corpus still needs a real Voyage API key unless the
query embedder is local too. A real gap found when eval-ci failed in CI with
"VOYAGE_API_KEY must be set" despite the corpus never needing re-embedding --
see docs/decisions/009. Both fixtures are loaded (not just the local one) so
anyone with a Voyage key can still exercise the production embedder path
locally against the same small dataset.
"""

import sys
from pathlib import Path

from filings_rag.config import get_settings
from filings_rag.db import get_connection, run_migrations

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
FIXTURE_PATH = FIXTURES_DIR / "eval_seed.sql"
LOCAL_FIXTURE_PATH = FIXTURES_DIR / "eval_seed_local.sql"


def main() -> int:
    settings = get_settings()
    conn = get_connection(settings)
    try:
        run_migrations(conn)
        conn.execute(FIXTURE_PATH.read_text())
        conn.execute(LOCAL_FIXTURE_PATH.read_text())
        conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
