"""Tests for the EDGAR client. No network: SEC is faked with httpx.MockTransport."""

from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pytest

from filings_rag.ingest.edgar import EdgarClient, latest_per_fiscal_year, parse_10k_rows
from filings_rag.ingest.models import Company

USER_AGENT = "FilingsRAG Test test@example.com"
APPLE = Company(ticker="AAPL", cik=320193, name="Apple")


class FakeClock:
    """Time that only moves when someone sleeps, so timing tests are instant and exact."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def filings_table(rows: list[tuple[str, str, str, str, str]]) -> dict[str, list[str]]:
    """Build EDGAR's column-oriented table from (form, accession, filed, period, doc) rows."""
    keys = ["form", "accessionNumber", "filingDate", "reportDate", "primaryDocument"]
    return {key: [row[i] for row in rows] for i, key in enumerate(keys)}


def make_client(
    tmp_path: Path, handler: Callable[[httpx.Request], httpx.Response], clock: FakeClock
) -> EdgarClient:
    return EdgarClient(
        user_agent=USER_AGENT,
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        clock=clock,
        sleep=clock.sleep,
    )


# --- client setup and HTTP behavior ---
# (RateLimiter and the retry loop itself are tested directly in test_http_retry.py;
# the tests below cover EdgarClient's own logic: filing selection, caching, and that
# it actually uses the shared retry helper end to end.)


@pytest.mark.parametrize("user_agent", [None, "", "FilingsRAG no-email"])
def test_requires_user_agent_with_email(tmp_path: Path, user_agent: str | None) -> None:
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        EdgarClient(user_agent=user_agent, cache_dir=tmp_path)


def test_sends_user_agent_header(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["User-Agent"])
        return httpx.Response(200, json={"filings": {"recent": filings_table([])}})

    make_client(tmp_path, handler, FakeClock()).list_10k_filings(APPLE, limit=3)
    assert seen == [USER_AGENT]


def test_retries_on_server_error_then_succeeds(tmp_path: Path) -> None:
    statuses = iter([503, 503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), json={"filings": {"recent": filings_table([])}})

    clock = FakeClock()
    make_client(tmp_path, handler, clock).list_10k_filings(APPLE, limit=3)
    # Exponential backoff: 1s, then 2s. (The limiter never needs to sleep because the
    # backoff sleeps already leave more than 1/8 s between requests.)
    assert clock.sleeps == [1.0, 2.0]


def test_honors_retry_after_header(tmp_path: Path) -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json={"filings": {"recent": filings_table([])}}),
        ]
    )
    clock = FakeClock()
    make_client(tmp_path, lambda _: next(responses), clock).list_10k_filings(APPLE, limit=3)
    assert clock.sleeps == [7.0]


def test_gives_up_after_max_attempts(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    with pytest.raises(httpx.HTTPStatusError):
        make_client(tmp_path, handler, FakeClock()).list_10k_filings(APPLE, limit=3)
    assert calls == 4


def test_does_not_retry_not_found(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        make_client(tmp_path, handler, FakeClock()).list_10k_filings(APPLE, limit=3)
    assert calls == 1


def test_retries_timeouts(tmp_path: Path) -> None:
    outcomes = iter(["timeout", "ok"])

    def handler(request: httpx.Request) -> httpx.Response:
        if next(outcomes) == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"filings": {"recent": filings_table([])}})

    clock = FakeClock()
    make_client(tmp_path, handler, clock).list_10k_filings(APPLE, limit=3)
    assert clock.sleeps == [1.0]


# --- choosing filings ---

RECENT = filings_table(
    [
        ("10-K", "0000320193-25-000079", "2025-10-31", "2025-09-27", "aapl-20250927.htm"),
        ("10-Q", "0000320193-25-000073", "2025-08-01", "2025-06-28", "aapl-20250628.htm"),
        ("10-K/A", "0000320193-25-000010", "2025-01-15", "2024-09-28", "amend.htm"),
        ("10-K", "0000320193-24-000123", "2024-11-01", "2024-09-28", "aapl-20240928.htm"),
    ]
)
OLDER_PAGE = filings_table(
    [
        ("10-K", "0000320193-23-000106", "2023-11-03", "2023-09-30", "aapl-20230930.htm"),
        ("10-K", "0000320193-22-000108", "2022-10-28", "2022-09-24", "aapl-20220924.htm"),
    ]
)


def test_parse_10k_rows_excludes_other_forms_and_amendments() -> None:
    filings = parse_10k_rows(RECENT, APPLE)
    assert [f.accession_number for f in filings] == [
        "0000320193-25-000079",
        "0000320193-24-000123",
    ]
    first = filings[0]
    assert first.fiscal_year == 2025
    assert first.report_date == date(2025, 9, 27)
    assert first.url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
    )


def test_latest_per_fiscal_year_keeps_newest_filing_per_year() -> None:
    refiled = filings_table(
        [
            ("10-K", "0000000001-24-000001", "2024-11-01", "2024-09-28", "a.htm"),
            ("10-K", "0000000001-24-000009", "2024-12-01", "2024-09-28", "b.htm"),
            ("10-K", "0000000001-23-000001", "2023-11-01", "2023-09-30", "c.htm"),
        ]
    )
    chosen = latest_per_fiscal_year(parse_10k_rows(refiled, APPLE), limit=3)
    assert [(f.fiscal_year, f.primary_document) for f in chosen] == [
        (2024, "b.htm"),
        (2023, "c.htm"),
    ]


def test_fetches_older_submission_pages_when_recent_is_not_enough(tmp_path: Path) -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path.endswith("CIK0000320193.json"):
            files = [{"name": "CIK0000320193-submissions-001.json"}]
            return httpx.Response(200, json={"filings": {"recent": RECENT, "files": files}})
        return httpx.Response(200, json=OLDER_PAGE)

    filings = make_client(tmp_path, handler, FakeClock()).list_10k_filings(APPLE, limit=3)
    assert [f.fiscal_year for f in filings] == [2025, 2024, 2023]
    assert requested == [
        "/submissions/CIK0000320193.json",
        "/submissions/CIK0000320193-submissions-001.json",
    ]


def test_skips_older_pages_when_recent_is_enough(tmp_path: Path) -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        files = [{"name": "CIK0000320193-submissions-001.json"}]
        return httpx.Response(200, json={"filings": {"recent": RECENT, "files": files}})

    filings = make_client(tmp_path, handler, FakeClock()).list_10k_filings(APPLE, limit=2)
    assert len(filings) == 2
    assert len(requested) == 1


# --- downloading ---


def test_download_caches_on_disk(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"<html>10-K</html>")

    client = make_client(tmp_path, handler, FakeClock())
    filing = parse_10k_rows(RECENT, APPLE)[0]

    path = client.download_filing(filing)
    again = client.download_filing(filing)

    assert path == again == tmp_path / "AAPL" / "0000320193-25-000079.htm"
    assert path.read_bytes() == b"<html>10-K</html>"
    assert calls == 1  # second call served from cache
    assert not path.with_suffix(".part").exists()
