"""SEC EDGAR client: polite downloading of 10-K filings.

"Polite" means what SEC's fair-access policy asks for:
- a User-Agent with a name and contact email on every request,
- staying under the rate limit (we use 8 req/s, SEC allows ~10),
- not re-downloading what we already have (filings are cached on disk).
"""

import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from filings_rag.config import Settings
from filings_rag.ingest.models import Company, FilingRef

SUBMISSIONS_BASE = "https://data.sec.gov/submissions/"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/"

# 429 = too many requests, 5xx = server trouble. Both are usually temporary.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RateLimiter:
    """Ensures at least 1/max_per_second seconds between calls to `wait()`.

    A fixed gap between requests is enough here because we download one file at a
    time; a token bucket would only matter if we allowed bursts or concurrency.
    `clock` and `sleep` are injectable so tests can use a fake clock.
    """

    def __init__(
        self,
        max_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._next_allowed = float("-inf")

    def wait(self) -> None:
        now = self._clock()
        if now < self._next_allowed:
            self._sleep(self._next_allowed - now)
            now = self._next_allowed
        self._next_allowed = now + self._interval


class EdgarClient:
    def __init__(
        self,
        user_agent: str | None,
        cache_dir: Path,
        max_requests_per_second: float = 8.0,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC_USER_AGENT must be set in .env with a name and contact email, e.g. "
                '"FilingsRAG Jane Doe jane@example.com" (required by SEC\'s fair-access policy).'
            )
        self._cache_dir = cache_dir
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._limiter = RateLimiter(max_requests_per_second, clock=clock, sleep=sleep)
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout_seconds,
            transport=transport,
            follow_redirects=True,
        )

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> "EdgarClient":
        return cls(
            user_agent=settings.sec_user_agent,
            cache_dir=settings.raw_dir,
            max_requests_per_second=settings.sec_max_requests_per_second,
            max_attempts=settings.sec_max_attempts,
            backoff_seconds=settings.sec_backoff_seconds,
            timeout_seconds=settings.sec_timeout_seconds,
            **kwargs,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "EdgarClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- public API ---

    def list_10k_filings(self, company: Company, limit: int) -> list[FilingRef]:
        """Return the `limit` most recent 10-K filings for a company, newest first.

        The submissions JSON is fetched fresh every time (not cached) because new filings
        get added to it. Its "recent" block holds only the last ~1000 filings; banks file
        thousands of prospectuses a year, so older 10-Ks can live in extra pages that we
        fetch only if needed.
        """
        submissions = self._get(f"{SUBMISSIONS_BASE}CIK{company.cik:010d}.json").json()
        filings = parse_10k_rows(submissions["filings"]["recent"], company)
        for page in submissions["filings"].get("files", []):
            if len({f.fiscal_year for f in filings}) >= limit:
                break
            filings += parse_10k_rows(self._get(SUBMISSIONS_BASE + page["name"]).json(), company)
        return latest_per_fiscal_year(filings, limit)

    def download_filing(self, filing: FilingRef) -> Path:
        """Download a filing's primary HTML document, or return the cached copy.

        Caching forever is safe: a filing never changes once published (amendments get
        a new accession number).
        """
        path = self._cache_dir / filing.ticker / f"{filing.accession_number}.htm"
        if path.exists():
            return path
        content = self._get(filing.url).content
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file, then rename: an interrupted download never leaves a
        # half-written file that later runs would mistake for a valid cache entry.
        tmp = path.with_suffix(".part")
        tmp.write_bytes(content)
        tmp.replace(path)
        return path

    # --- internals ---

    def _get(self, url: str) -> httpx.Response:
        """GET with rate limiting and retries (exponential backoff: 1s, 2s, 4s, ...)."""
        for attempt in range(self._max_attempts):
            is_last = attempt == self._max_attempts - 1
            self._limiter.wait()
            try:
                response = self._http.get(url)
            except httpx.TransportError:  # timeouts, dropped connections
                if is_last:
                    raise
                self._sleep(self._backoff_seconds * 2**attempt)
                continue
            if response.status_code in RETRYABLE_STATUS and not is_last:
                self._sleep(self._retry_delay(response, attempt))
                continue
            response.raise_for_status()  # 404 etc. aren't retried: they won't fix themselves
            return response
        raise AssertionError("unreachable")

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        # If SEC says how long to wait, do that; otherwise back off exponentially.
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            return float(retry_after)
        return self._backoff_seconds * 2**attempt


def parse_10k_rows(table: dict[str, list[Any]], company: Company) -> list[FilingRef]:
    """Turn EDGAR's column-oriented filings table into FilingRefs, keeping only 10-Ks.

    Exact match on "10-K" excludes amendments ("10-K/A"), which usually only patch a
    few items and would duplicate the original filing.
    """
    filings = []
    for i, form in enumerate(table["form"]):
        if form != "10-K":
            continue
        accession = table["accessionNumber"][i]
        document = table["primaryDocument"][i]
        report_date = date.fromisoformat(table["reportDate"][i])
        filings.append(
            FilingRef(
                ticker=company.ticker,
                cik=company.cik,
                company=company.name,
                form=form,
                accession_number=accession,
                filing_date=date.fromisoformat(table["filingDate"][i]),
                report_date=report_date,
                # Year of the period end: e.g. NVIDIA's FY2025 ends 2025-01-26 -> 2025.
                fiscal_year=report_date.year,
                primary_document=document,
                url=f"{ARCHIVES_BASE}{company.cik}/{accession.replace('-', '')}/{document}",
            )
        )
    return filings


def latest_per_fiscal_year(filings: list[FilingRef], limit: int) -> list[FilingRef]:
    """Keep one filing per fiscal year (the latest filed), newest years first."""
    by_year: dict[int, FilingRef] = {}
    for filing in sorted(filings, key=lambda f: f.filing_date):
        by_year[filing.fiscal_year] = filing
    return [by_year[year] for year in sorted(by_year, reverse=True)[:limit]]
