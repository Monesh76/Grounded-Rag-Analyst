"""Pulls a company and a fiscal year out of a question, when present, so
retrieval can narrow its search with a WHERE clause instead of relying on
similarity alone to find the right company.
"""

import re

from filings_rag.ingest.companies import COMPANIES
from filings_rag.retrieve.models import Filters

_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def extract_filters(question: str) -> Filters:
    return Filters(ticker=_extract_ticker(question), fiscal_year=_extract_year(question))


def _extract_ticker(question: str) -> str | None:
    # Longest name first: "Bank of America" should win over a shorter substring
    # match before a shorter one could grab part of it.
    for company in sorted(COMPANIES, key=lambda c: -len(c.name)):
        if re.search(rf"\b{re.escape(company.name)}\b", question, re.IGNORECASE):
            return company.ticker
    for company in COMPANIES:
        if re.search(rf"\b{re.escape(company.ticker)}\b", question):
            return company.ticker
    return None


def _extract_year(question: str) -> int | None:
    match = _YEAR_RE.search(question)
    return int(match.group(1)) if match else None
