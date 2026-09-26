"""Tests for the loader. Needs a real Postgres; uses a fake Embedder (no real
Voyage/local calls -- those are already covered in test_embed.py) so this stays
fast and free while still exercising the real chunk -> embed -> upsert path.
"""

from pathlib import Path

import psycopg
import pytest

from filings_rag.config import Settings, get_settings
from filings_rag.db import get_connection, run_migrations
from filings_rag.ingest.load import load_all
from filings_rag.ingest.models import FilingRef, Manifest, ManifestEntry, ParsedFiling, Section

pytestmark = pytest.mark.integration


class FakeEmbedder:
    """Deterministic, free, instant -- sized to match the chunks table's vector(1024)."""

    dimensions = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7)] * self.dimensions for t in texts]


FILING = FilingRef(
    ticker="TEST",
    cik=1,
    company="Test Co",
    form="10-K",
    accession_number="0000000001-24-000001",
    filing_date="2024-11-01",
    report_date="2024-09-28",
    fiscal_year=2024,
    primary_document="test.htm",
    url="https://example.com/test.htm",
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Real Postgres (via database_url), but manifest/parsed files under tmp_path."""
    s = get_settings().model_copy(update={"data_dir": tmp_path})
    s.data_dir.mkdir(exist_ok=True)
    s.parsed_dir.mkdir(exist_ok=True)

    section = Section(item="1", title="Business", page=1, text="We make things. " * 20)
    (s.parsed_dir / "TEST_2024.json").write_text(
        ParsedFiling(filing=FILING, sections=[section]).model_dump_json()
    )
    entry = ManifestEntry(**FILING.model_dump(), raw_path="raw/TEST/x.htm", sha256="deadbeef")
    s.manifest_path.write_text(
        Manifest(generated_at="2024-11-01T00:00:00Z", filings=[entry]).model_dump_json()
    )
    return s


@pytest.fixture(autouse=True)
def clean_chunks_table() -> None:
    # Doesn't assume test_db.py (or anything else) already created the schema --
    # run_migrations is idempotent, so this is safe however the test files are ordered.
    try:
        conn = get_connection(get_settings())
    except psycopg.OperationalError as exc:
        pytest.fail(f"Cannot reach Postgres. Run `docker compose up -d db`.\n{exc}")
    run_migrations(conn)
    with conn:
        conn.execute("DELETE FROM chunks WHERE ticker IN ('TEST', 'GONE')")
    conn.close()


def test_loads_chunks_for_each_filing_in_manifest(settings: Settings) -> None:
    summary = load_all(settings, embedder=FakeEmbedder())
    assert summary["filings_loaded"] == 1
    assert summary["chunks_loaded"] > 0
    assert summary["total_chunks_in_db"] >= summary["chunks_loaded"]


def test_rerunning_does_not_duplicate_rows(settings: Settings) -> None:
    first = load_all(settings, embedder=FakeEmbedder())
    second = load_all(settings, embedder=FakeEmbedder())
    assert second["chunks_loaded"] == first["chunks_loaded"]
    assert second["total_chunks_in_db"] == first["total_chunks_in_db"]


def test_rerunning_updates_changed_text(settings: Settings) -> None:
    load_all(settings, embedder=FakeEmbedder())

    section = Section(item="1", title="Business", page=1, text="Updated content. " * 20)
    (settings.parsed_dir / "TEST_2024.json").write_text(
        ParsedFiling(filing=FILING, sections=[section]).model_dump_json()
    )
    load_all(settings, embedder=FakeEmbedder())

    conn = psycopg.connect(settings.database_url)
    (text,) = conn.execute("SELECT text FROM chunks WHERE ticker = 'TEST' LIMIT 1").fetchone()
    conn.close()
    assert "Updated content" in text


def test_skips_manifest_entries_missing_parsed_file(settings: Settings, tmp_path: Path) -> None:
    missing = FILING.model_copy(update={"ticker": "GONE", "accession_number": "0000000002"})
    entry = ManifestEntry(**missing.model_dump(), raw_path="raw/GONE/x.htm", sha256="dead")
    existing_entry = ManifestEntry(
        **FILING.model_dump(), raw_path="raw/TEST/x.htm", sha256="deadbeef"
    )
    settings.manifest_path.write_text(
        Manifest(
            generated_at="2024-11-01T00:00:00Z", filings=[entry, existing_entry]
        ).model_dump_json()
    )

    summary = load_all(settings, embedder=FakeEmbedder())
    assert summary["filings_loaded"] == 1  # GONE skipped, TEST loaded
