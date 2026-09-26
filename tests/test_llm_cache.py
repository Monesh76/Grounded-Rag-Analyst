from pathlib import Path

from evals.llm_cache import CachingLLMProvider
from filings_rag.generate.models import LLMResponse


class FakeProvider:
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=f"answer #{self.calls}",
            model=self.model,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
        )


def test_second_identical_call_is_served_from_cache(tmp_path: Path) -> None:
    inner = FakeProvider()
    cached = CachingLLMProvider(inner, cache_dir=tmp_path)

    first = cached.complete("system", "question")
    second = cached.complete("system", "question")

    assert inner.calls == 1  # the real provider was only called once
    assert first.text == second.text == "answer #1"
    assert cached.hits == 1
    assert cached.misses == 1


def test_different_user_text_is_not_a_cache_hit(tmp_path: Path) -> None:
    inner = FakeProvider()
    cached = CachingLLMProvider(inner, cache_dir=tmp_path)

    cached.complete("system", "question A")
    cached.complete("system", "question B")

    assert inner.calls == 2
    assert cached.misses == 2


def test_different_system_prompt_is_not_a_cache_hit(tmp_path: Path) -> None:
    inner = FakeProvider()
    cached = CachingLLMProvider(inner, cache_dir=tmp_path)

    cached.complete("system A", "question")
    cached.complete("system B", "question")

    assert inner.calls == 2


def test_cache_persists_across_provider_instances(tmp_path: Path) -> None:
    inner = FakeProvider()
    CachingLLMProvider(inner, cache_dir=tmp_path).complete("system", "question")

    inner2 = FakeProvider()
    result = CachingLLMProvider(inner2, cache_dir=tmp_path).complete("system", "question")

    assert inner2.calls == 0  # served entirely from disk, the new instance never called
    assert result.text == "answer #1"
