"""Shared helpers for polite outbound HTTP: fixed-rate spacing and retry with
backoff. Used by the EDGAR client and the Voyage embedder, which both need the
same shape (space out requests, retry transient failures, honor `Retry-After`,
don't retry a client error that won't fix itself).
"""

import time
from collections.abc import Callable

import httpx

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


def request_with_retry(
    send: Callable[[], httpx.Response],
    max_attempts: int,
    backoff_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """Call `send()` (which should do its own rate-limit wait each time, since a
    retry is itself a new request), retrying on transient failures with
    exponential backoff (1s, 2s, 4s, ...), honoring `Retry-After` when a server
    sends one. A non-retryable error (e.g. 404, 401) raises immediately.
    """
    for attempt in range(max_attempts):
        is_last = attempt == max_attempts - 1
        try:
            response = send()
        except httpx.TransportError:  # timeouts, dropped connections
            if is_last:
                raise
            sleep(backoff_seconds * 2**attempt)
            continue
        if response.status_code in RETRYABLE_STATUS and not is_last:
            sleep(_retry_delay(response, backoff_seconds, attempt))
            continue
        response.raise_for_status()
        return response
    raise AssertionError("unreachable")


def _retry_delay(response: httpx.Response, backoff_seconds: float, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After", "")
    if retry_after.isdigit():
        return float(retry_after)
    return backoff_seconds * 2**attempt
