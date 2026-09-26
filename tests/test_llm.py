"""Tests for the LLM providers. No network: each provider's `client` is a fake
object shaped like the real SDK client, so from_settings/the real SDKs are
never exercised here (that's verified manually against real APIs instead).
"""

from types import SimpleNamespace

import pytest

from filings_rag.generate.llm import ClaudeProvider, OpenAICompatibleProvider


class FakeAnthropicClient:
    def __init__(
        self, text: str = "answer", input_tokens: int = 10, output_tokens: int = 5
    ) -> None:
        self.messages = SimpleNamespace(create=self._create)
        self._text = text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.last_call: dict | None = None

    def _create(self, **kwargs):
        self.last_call = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)],
            usage=SimpleNamespace(
                input_tokens=self._input_tokens, output_tokens=self._output_tokens
            ),
        )


class FakeOpenAIClient:
    def __init__(
        self, text: str = "answer", prompt_tokens: int = 10, completion_tokens: int = 5
    ) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._text = text
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.last_call: dict | None = None

    def _create(self, **kwargs):
        self.last_call = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))],
            usage=SimpleNamespace(
                prompt_tokens=self._prompt_tokens, completion_tokens=self._completion_tokens
            ),
        )


# --- ClaudeProvider ---


def test_claude_returns_text_and_usage() -> None:
    provider = ClaudeProvider(FakeAnthropicClient(text="Apple's revenue was..."), "claude-sonnet-5")
    response = provider.complete("system prompt", "user question")
    assert response.text == "Apple's revenue was..."
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.model == "claude-sonnet-5"


def test_claude_passes_system_and_user_correctly() -> None:
    client = FakeAnthropicClient()
    ClaudeProvider(client, "claude-sonnet-5").complete("be helpful", "what is x?")
    assert client.last_call["system"] == "be helpful"
    assert client.last_call["messages"] == [{"role": "user", "content": "what is x?"}]


def test_claude_computes_cost_from_known_pricing() -> None:
    provider = ClaudeProvider(
        FakeAnthropicClient(input_tokens=1_000_000, output_tokens=1_000_000), "claude-sonnet-5"
    )
    response = provider.complete("s", "u")
    assert response.cost_usd == pytest.approx(3.00 + 15.00)


def test_claude_cost_is_zero_for_unknown_model() -> None:
    provider = ClaudeProvider(FakeAnthropicClient(), "some-future-model")
    response = provider.complete("s", "u")
    assert response.cost_usd == 0.0


def test_claude_joins_multiple_text_blocks() -> None:
    client = FakeAnthropicClient()
    client.messages.create = lambda **kw: SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="part one. "),
            SimpleNamespace(type="text", text="part two."),
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    response = ClaudeProvider(client, "claude-sonnet-5").complete("s", "u")
    assert response.text == "part one. part two."


# --- OpenAICompatibleProvider ---


def test_openai_compatible_returns_text_and_usage() -> None:
    provider = OpenAICompatibleProvider(FakeOpenAIClient(text="the answer"), "gpt-4o-mini")
    response = provider.complete("system prompt", "user question")
    assert response.text == "the answer"
    assert response.input_tokens == 10
    assert response.output_tokens == 5


def test_openai_compatible_passes_system_and_user_as_messages() -> None:
    client = FakeOpenAIClient()
    OpenAICompatibleProvider(client, "gpt-4o-mini").complete("be helpful", "what is x?")
    assert client.last_call["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "what is x?"},
    ]


def test_openai_compatible_sends_extra_headers_when_given() -> None:
    client = FakeOpenAIClient()
    provider = OpenAICompatibleProvider(
        client, "gpt-4o-mini", extra_headers={"X-Title": "FilingsRAG"}
    )
    provider.complete("s", "u")
    assert client.last_call["extra_headers"] == {"X-Title": "FilingsRAG"}


def test_openai_compatible_handles_missing_usage() -> None:
    client = FakeOpenAIClient()
    client.chat.completions.create = lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))], usage=None
    )
    response = OpenAICompatibleProvider(client, "gpt-4o-mini").complete("s", "u")
    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.cost_usd == 0.0
