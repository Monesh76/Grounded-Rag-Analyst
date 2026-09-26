"""Caches LLM completions on disk by a hash of (model, system, user), so
re-running eval-full during development doesn't re-pay for a question whose
answer hasn't changed. Wraps any LLMProvider; the eval harness's own judge
calls (DeepEval) aren't covered by this -- a separate, documented limitation.
"""

import hashlib
from pathlib import Path

from filings_rag.generate.llm import LLMProvider
from filings_rag.generate.models import LLMResponse

DEFAULT_CACHE_DIR = Path(".cache/llm")


class CachingLLMProvider:
    def __init__(self, inner: LLMProvider, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = inner.model
        self.hits = 0
        self.misses = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        key = hashlib.sha256(f"{self.model}\n{system}\n{user}".encode()).hexdigest()
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            return LLMResponse.model_validate_json(path.read_text())

        self.misses += 1
        response = self._inner.complete(system, user)
        path.write_text(response.model_dump_json())
        return response
