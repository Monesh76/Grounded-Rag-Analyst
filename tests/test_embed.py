"""Tests for the Embedder implementations. No network and no real model downloads:
Voyage is faked with httpx.MockTransport, and the local embedder is given a fake
model object rather than a real sentence-transformers download.
"""

import json
from collections.abc import Callable

import httpx
import pytest

from filings_rag.embed import Embedder, LocalEmbedder, VoyageEmbedder

API_KEY = "test-key"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_voyage(
    handler: Callable[[httpx.Request], httpx.Response], batch_size: int = 100
) -> VoyageEmbedder:
    clock = FakeClock()
    return VoyageEmbedder(
        api_key=API_KEY,
        model="voyage-finance-2",
        dimensions=1024,
        batch_size=batch_size,
        transport=httpx.MockTransport(handler),
        clock=clock,
        sleep=clock.sleep,
    )


def voyage_response(texts: list[str]) -> httpx.Response:
    # A tiny fake embedding per text so we can check order without real vectors.
    data = [{"embedding": [float(i)], "index": i} for i in range(len(texts))]
    return httpx.Response(200, json={"data": data, "usage": {"total_tokens": len(texts) * 10}})


# --- protocol conformance ---


def test_both_implementations_satisfy_the_protocol() -> None:
    voyage = make_voyage(lambda r: voyage_response(["x"]))
    local = LocalEmbedder(model=_FakeModel())
    assert isinstance(voyage, Embedder)
    assert isinstance(local, Embedder)


# --- VoyageEmbedder ---


def test_requires_api_key() -> None:
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        VoyageEmbedder(api_key=None, model="voyage-finance-2", dimensions=1024)


def test_sends_auth_header_and_model() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"auth": request.headers["Authorization"], "body": json.loads(request.content)})
        return voyage_response(["a", "b"])

    make_voyage(handler).embed(["a", "b"])
    assert seen[0]["auth"] == f"Bearer {API_KEY}"
    assert seen[0]["body"] == {"input": ["a", "b"], "model": "voyage-finance-2"}


def test_returns_embeddings_in_input_order_even_if_api_reorders() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Reverse the "index" ordering to simulate an API that doesn't guarantee order.
        data = [{"embedding": [1.0], "index": 1}, {"embedding": [0.0], "index": 0}]
        return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 2}})

    result = make_voyage(handler).embed(["first", "second"])
    assert result == [[0.0], [1.0]]


def test_batches_large_inputs() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        calls.append(len(texts))
        return voyage_response(texts)

    embedder = make_voyage(handler, batch_size=2)
    result = embedder.embed(["a", "b", "c", "d", "e"])
    assert calls == [2, 2, 1]
    assert len(result) == 5


def test_tracks_total_tokens_used_for_cost_estimate() -> None:
    embedder = make_voyage(lambda r: voyage_response(["a", "b"]))
    embedder.embed(["a", "b"])
    assert embedder.total_tokens_used == 20


def test_retries_on_server_error() -> None:
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 503:
            return httpx.Response(503)
        return voyage_response(["a"])

    result = make_voyage(handler).embed(["a"])
    assert result == [[0.0]]


# --- LocalEmbedder ---


class _FakeModel:
    """Stands in for a sentence-transformers model, without downloading one."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.encode_calls: list[dict] = []

    def get_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str], batch_size: int, show_progress_bar: bool):
        import numpy as np

        self.encode_calls.append({"texts": texts, "batch_size": batch_size})
        return np.array([[float(len(t))] * self._dim for t in texts])


def test_local_embedder_reports_model_dimensions() -> None:
    embedder = LocalEmbedder(model=_FakeModel(dim=384))
    assert embedder.dimensions == 384


def test_local_embedder_returns_one_vector_per_text() -> None:
    embedder = LocalEmbedder(model=_FakeModel())
    vectors = embedder.embed(["hi", "a longer bit of text"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 4


def test_local_embedder_passes_batch_size_through() -> None:
    model = _FakeModel()
    embedder = LocalEmbedder(model=model, batch_size=8)
    embedder.embed(["a", "b"])
    assert model.encode_calls[0]["batch_size"] == 8
