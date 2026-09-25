"""Pydantic models for ingestion outputs (filing metadata, sections, manifest)."""

from datetime import date, datetime

from pydantic import BaseModel


class Company(BaseModel):
    ticker: str
    cik: int  # SEC's Central Index Key, a permanent company id
    name: str


class FilingRef(BaseModel):
    """Metadata for one filing, as listed in EDGAR's submissions JSON."""

    ticker: str
    cik: int
    company: str
    form: str
    accession_number: str  # e.g. "0000320193-24-000123"; unique and immutable per filing
    filing_date: date
    report_date: date  # end of the fiscal period the filing covers
    fiscal_year: int
    primary_document: str  # the main HTML file inside the filing
    url: str


class Section(BaseModel):
    item: str  # "1", "1A", "7", ...
    title: str
    page: int | None  # approximate page number, counted from page breaks
    text: str


class ParsedFiling(BaseModel):
    filing: FilingRef
    sections: list[Section]


class Chunk(BaseModel):
    # Deterministic id (ticker_fiscalyear_item_index), e.g. "AAPL_2025_7_3": makes
    # citations self-describing and re-running the loader idempotent (upsert by id).
    id: str
    doc_id: str  # accession number of the filing this chunk came from
    ticker: str
    company: str
    fiscal_year: int
    item: str
    section_title: str
    page: int  # approximate: the page the section started on, not this chunk specifically
    chunk_index: int
    text: str


class ManifestEntry(FilingRef):
    raw_path: str
    sha256: str  # hash of the raw HTML, so a re-download can be checked against it


class Manifest(BaseModel):
    generated_at: datetime
    filings: list[ManifestEntry]
