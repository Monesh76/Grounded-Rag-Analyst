"""Tests for the shared rate-limiter and retry helper. No network: httpx errors and
responses are constructed directly rather than going through a real transport.
"""

import httpx
import pytest

from filings_rag.http_retry import RateLimiter, request_with_retry

URL = "https://example.com/thing"


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


# --- RateLimiter ---


def test_rate_limiter_spaces_calls() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_per_second=8, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.wait()
    assert clock.sleeps == [pytest.approx(0.125), pytest.approx(0.125)]


def test_rate_limiter_does_not_sleep_when_calls_are_slow() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_per_second=8, clock=clock, sleep=clock.sleep)
    limiter.wait()
    clock.now += 1.0  # plenty of time passes between requests
    limiter.wait()
    assert clock.sleeps == []


# --- request_with_retry ---


def test_returns_response_on_first_success() -> None:
    clock = FakeClock()
    response = request_with_retry(
        lambda: httpx.Response(200, request=httpx.Request("GET", URL)),
        max_attempts=4,
        backoff_seconds=1.0,
        sleep=clock.sleep,
    )
    assert response.status_code == 200
    assert clock.sleeps == []


def test_retries_on_server_error_then_succeeds() -> None:
    statuses = iter([503, 503, 200])
    clock = FakeClock()
    response = request_with_retry(
        lambda: httpx.Response(next(statuses), request=httpx.Request("GET", URL)),
        max_attempts=4,
        backoff_seconds=1.0,
        sleep=clock.sleep,
    )
    assert response.status_code == 200
    assert clock.sleeps == [1.0, 2.0]  # exponential backoff


def test_honors_retry_after_header() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "7"}, request=httpx.Request("GET", URL)),
            httpx.Response(200, request=httpx.Request("GET", URL)),
        ]
    )
    clock = FakeClock()
    request_with_retry(
        lambda: next(responses), max_attempts=4, backoff_seconds=1.0, sleep=clock.sleep
    )
    assert clock.sleeps == [7.0]


def test_gives_up_after_max_attempts() -> None:
    calls = 0

    def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, request=httpx.Request("GET", URL))

    with pytest.raises(httpx.HTTPStatusError):
        request_with_retry(send, max_attempts=4, backoff_seconds=1.0, sleep=lambda _: None)
    assert calls == 4


def test_does_not_retry_client_error() -> None:
    calls = 0

    def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, request=httpx.Request("GET", URL))

    with pytest.raises(httpx.HTTPStatusError):
        request_with_retry(send, max_attempts=4, backoff_seconds=1.0, sleep=lambda _: None)
    assert calls == 1


def test_retries_transport_errors() -> None:
    outcomes = iter(["timeout", "ok"])

    def send() -> httpx.Response:
        if next(outcomes) == "timeout":
            raise httpx.ReadTimeout("slow", request=httpx.Request("GET", URL))
        return httpx.Response(200, request=httpx.Request("GET", URL))

    clock = FakeClock()
    request_with_retry(send, max_attempts=4, backoff_seconds=1.0, sleep=clock.sleep)
    assert clock.sleeps == [1.0]
