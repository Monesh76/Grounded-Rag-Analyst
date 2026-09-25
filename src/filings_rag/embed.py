"""Embedding: one Protocol, two implementations (PLAN.md's "behind an interface"
requirement, so nothing downstream cares which one is in use).

- VoyageEmbedder: hosted API, costs money, needs VOYAGE_API_KEY.
- LocalEmbedder: sentence-transformers, runs on CPU, no API cost.
"""

import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

import httpx

from filings_rag.config import Settings
from filings_rag.http_retry import RateLimiter, request_with_retry

VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order as the input."""
        ...


def _batched(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        dimensions: int,
        batch_size: int = 100,
        max_requests_per_second: float = 4.0,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
        timeout_seconds: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("VOYAGE_API_KEY must be set in .env to use the Voyage embedder.")
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._limiter = RateLimiter(max_requests_per_second, clock=clock, sleep=sleep)
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )
        self.total_tokens_used = 0  # for the cost estimate `make load` prints

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> "VoyageEmbedder":
        return cls(
            api_key=settings.voyage_api_key,
            model=settings.voyage_model,
            dimensions=1024,  # voyage-finance-2's native output size
            batch_size=settings.voyage_batch_size,
            max_requests_per_second=settings.voyage_max_requests_per_second,
            max_attempts=settings.voyage_max_attempts,
            backoff_seconds=settings.voyage_backoff_seconds,
            timeout_seconds=settings.voyage_timeout_seconds,
            **kwargs,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "VoyageEmbedder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batched(texts, self._batch_size):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        def send() -> httpx.Response:
            self._limiter.wait()
            return self._http.post(
                VOYAGE_EMBEDDINGS_URL, json={"input": list(batch), "model": self._model}
            )

        response = request_with_retry(
            send, self._max_attempts, self._backoff_seconds, sleep=self._sleep
        )
        body = response.json()
        self.total_tokens_used += body.get("usage", {}).get("total_tokens", 0)
        # Voyage's docs don't guarantee response order matches input order; sort by
        # the "index" field it returns alongside each embedding to be safe.
        ordered = sorted(body["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in ordered]


class LocalEmbedder:
    """Wraps a sentence-transformers model. `model` is injectable (any object with
    `.encode()` and `.get_sentence_embedding_dimension()`) so tests don't need to
    download a real model -- only `from_settings` and the default path do that."""

    def __init__(self, model: Any, batch_size: int = 32) -> None:
        self._model = model
        self._batch_size = batch_size

    @classmethod
    def from_settings(cls, settings: Settings) -> "LocalEmbedder":
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        model = SentenceTransformer(settings.local_embedding_model)
        return cls(model, settings.local_embedding_batch_size)

    @property
    def dimensions(self) -> int:
        return self._model.get_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), batch_size=self._batch_size, show_progress_bar=False
        )
        return vectors.tolist()
