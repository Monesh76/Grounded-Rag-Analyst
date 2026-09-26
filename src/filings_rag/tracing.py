"""Langfuse tracing: one root span per `ask()` call, with child spans for
dense/keyword retrieval, rerank, and the LLM generation -- PLAN.md's P6 task.

Disabled (a harmless no-op) whenever LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY
aren't configured, the same pattern the LLM/Embedder providers already use for
their own API keys -- so `make test` and CI's default job stay network-free
without needing Langfuse credentials.

Uses manual `start_as_current_observation` spans rather than the `@observe`
decorator: this project builds its own `Langfuse` client from `settings`
(config.py's "no keys read outside config.py" convention) instead of the
env-var-based global client `@observe` relies on, so explicit spans are the
more predictable integration.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from filings_rag.config import Settings

_clients: dict[int, Any] = {}


def _client(settings: Settings) -> Any | None:
    key = id(settings)
    if key not in _clients:
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            from langfuse import Langfuse

            _clients[key] = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_base_url,
            )
        else:
            _clients[key] = None
    return _clients[key]


@contextmanager
def span(
    settings: Settings, name: str, as_type: str = "span", **input_kwargs: Any
) -> Iterator[Any]:
    """A tracing span if Langfuse is configured, otherwise yields None.
    Callers must guard follow-up calls with `update(obs, ...)`, not
    `obs.update(...)` directly, so the no-op case stays a no-op."""
    client = _client(settings)
    if client is None:
        yield None
        return
    with client.start_as_current_observation(name=name, as_type=as_type, **input_kwargs) as obs:
        yield obs


def update(obs: Any, **kwargs: Any) -> None:
    if obs is not None:
        obs.update(**kwargs)


def flush(settings: Settings) -> None:
    """Short-lived processes (CLI entrypoints, the eval harness) must flush
    explicitly before exit -- Langfuse batches and sends on its own schedule
    otherwise, which a process that's already exited never waits around for."""
    client = _client(settings)
    if client is not None:
        client.flush()
