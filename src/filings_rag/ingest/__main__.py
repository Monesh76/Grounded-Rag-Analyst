"""Ingestion entry point: `python -m filings_rag.ingest`.

For each company in COMPANIES: fetch the 3 most recent 10-Ks, download and cache
the HTML, parse it into sections, and write data/manifest.json. One company or
filing failing (a network hiccup, an unparseable filing) does not stop the rest --
failures are collected and reported at the end instead, next to the section-count
summary PLAN.md asks for.
"""

import hashlib
import sys
from datetime import UTC, datetime

from filings_rag.config import get_settings
from filings_rag.ingest.companies import COMPANIES
from filings_rag.ingest.edgar import EdgarClient
from filings_rag.ingest.models import Manifest, ManifestEntry, ParsedFiling
from filings_rag.ingest.parse import parse_filing_html

# PLAN.md's "done when": every filing should have these four items detected.
EXPECTED_ITEMS = {"1", "1A", "7", "8"}


def main() -> int:
    settings = get_settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)

    entries: list[ManifestEntry] = []
    rows: list[tuple[str, int, list[str]]] = []  # (ticker, fiscal_year, items found)
    errors: list[str] = []

    with EdgarClient.from_settings(settings) as client:
        for company in COMPANIES:
            try:
                filings = client.list_10k_filings(company, limit=settings.filings_per_company)
            except Exception as exc:  # noqa: BLE001 -- one bad company shouldn't stop the rest
                errors.append(f"{company.ticker}: could not list filings ({exc})")
                continue

            for filing in filings:
                try:
                    path = client.download_filing(filing)
                    html = path.read_bytes()
                    sections = parse_filing_html(html)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{filing.ticker} FY{filing.fiscal_year}: {exc}")
                    continue

                parsed_path = settings.parsed_dir / f"{filing.ticker}_{filing.fiscal_year}.json"
                parsed_path.write_text(
                    ParsedFiling(filing=filing, sections=sections).model_dump_json(indent=2)
                )

                entries.append(
                    ManifestEntry(
                        **filing.model_dump(),
                        raw_path=str(path.relative_to(settings.data_dir)),
                        sha256=hashlib.sha256(html).hexdigest(),
                    )
                )
                rows.append((filing.ticker, filing.fiscal_year, sorted(s.item for s in sections)))

    manifest = Manifest(generated_at=datetime.now(UTC), filings=entries)
    settings.manifest_path.write_text(manifest.model_dump_json(indent=2))

    _print_summary(rows, errors, settings.manifest_path)
    return 1 if errors else 0


def _print_summary(
    rows: list[tuple[str, int, list[str]]], errors: list[str], manifest_path: object
) -> None:
    print(f"\n{'Ticker':<8}{'FY':<6}{'Items':<30}Missing")
    print("-" * 70)
    for ticker, fiscal_year, items in sorted(rows):
        missing = sorted(EXPECTED_ITEMS - set(items))
        flag = f"  MISSING: {', '.join(missing)}" if missing else ""
        print(f"{ticker:<8}{fiscal_year:<6}{', '.join(items):<30}{flag}")

    print(f"\n{len(rows)} filings parsed, {len(errors)} errors.")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")

    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
