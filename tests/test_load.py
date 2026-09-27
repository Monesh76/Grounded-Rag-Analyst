"""Tests for the loader. Needs a real Postgres; uses a fake Embedder (no real
Voyage/local calls -- those are already covered in test_embed.py) so this stays
fast and free while still exercising the real chunk -> embed -> upsert path.
"""

from pathlib import Path

import psycopg
import pytest

from filings_rag.config import Settings, get_settings
from filings_rag.db import get_test_connection, run_migrations
from filings_rag.ingest.chunk import chunk_filing_fixed
from filings_rag.ingest.load import _upsert_sql, load_all
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
    """The separate test database (load_all reads settings.database_url, so we
    point that at test_database_url here), manifest/parsed files under tmp_path."""
    base = get_settings()
    s = base.model_copy(update={"data_dir": tmp_path, "database_url": base.test_database_url})
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
        conn = get_test_connection(get_settings())
    except psycopg.OperationalError as exc:
        pytest.fail(f"Cannot reach Postgres. Run `docker compose up -d db`.\n{exc}")
    run_migrations(conn)
    conn.execute("DELETE FROM chunks WHERE ticker IN ('TEST', 'GONE')")
    conn.execute("DELETE FROM chunks_fixed512 WHERE ticker IN ('TEST', 'GONE')")
    conn.commit()
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


def test_loads_into_chunks_fixed512_with_the_fixed_chunker(settings: Settings) -> None:
    summary = load_all(
        settings, embedder=FakeEmbedder(), table="chunks_fixed512", chunk_fn=chunk_filing_fixed
    )
    assert summary["chunks_loaded"] > 0

    conn = psycopg.connect(settings.database_url)
    (count,) = conn.execute("SELECT count(*) FROM chunks_fixed512 WHERE ticker = 'TEST'").fetchone()
    (chunks_count,) = conn.execute("SELECT count(*) FROM chunks WHERE ticker = 'TEST'").fetchone()
    conn.close()
    assert count == summary["chunks_loaded"]
    assert chunks_count == 0  # the production table is untouched


def test_upsert_sql_rejects_unknown_table() -> None:
    with pytest.raises(ValueError, match="Unknown chunk table"):
        _upsert_sql("drop_all")
