"""LLM access behind one protocol, three implementations:

- ClaudeProvider: Anthropic's SDK directly.
- OpenAICompatibleProvider: OpenAI's SDK, used both for the real OpenAI API and
  for OpenRouter (an OpenAI-compatible proxy that can route to many models,
  including cheap ones) -- same client shape, just a different base_url/model.

`llm_provider` in config picks which one `get_llm_provider()` builds. OpenRouter
is the default (cheap for everyday dev/testing); set it to "claude" or "openai"
in .env for final checks and production, per PLAN.md.
"""

from typing import Any, Protocol, runtime_checkable

from filings_rag.config import Settings
from filings_rag.generate.models import LLMResponse

# PLACEHOLDER prices ($ per million tokens, input/output) -- verify against each
# provider's current pricing page before trusting cost_usd. OpenRouter's price for
# a given model varies by provider/routing, so this is a rough estimate there.
_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
}


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = _PRICING.get(model)
    if prices is None:
        return 0.0
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


@runtime_checkable
class LLMProvider(Protocol):
    model: str

    def complete(self, system: str, user: str) -> LLMResponse: ...


class ClaudeProvider:
    """`client` is injectable (any object shaped like anthropic.Anthropic) so
    tests don't need a real API key or network call."""

    def __init__(
        self, client: Any, model: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> None:
        self._client = client
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @classmethod
    def from_settings(cls, settings: Settings) -> "ClaudeProvider":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in .env to use the Claude provider.")
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return cls(
            client, settings.anthropic_model, settings.llm_max_tokens, settings.llm_temperature
        )

    def complete(self, system: str, user: str) -> LLMResponse:
        # The installed SDK's Messages.create has no `temperature` param for this
        # model family (replaced by output_config.effort, a reasoning-effort
        # control -- a different axis, not a substitute for sampling temperature).
        # self._temperature is accepted for a consistent constructor across
        # providers but isn't used here; verified against the live SDK signature
        # rather than assumed.
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost_usd(self.model, input_tokens, output_tokens),
        )


class OpenAICompatibleProvider:
    """`client` is injectable (any object shaped like openai.OpenAI) so tests
    don't need a real API key or network call. Used for both the real OpenAI API
    and OpenRouter, which exposes the same chat-completions shape."""

    def __init__(
        self,
        client: Any,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._extra_headers = extra_headers or {}

    @classmethod
    def from_settings_openai(cls, settings: Settings) -> "OpenAICompatibleProvider":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env to use the OpenAI provider.")
        import openai

        client = openai.OpenAI(api_key=settings.openai_api_key)
        return cls(client, settings.openai_model, settings.llm_max_tokens, settings.llm_temperature)

    @classmethod
    def from_settings_openrouter(cls, settings: Settings) -> "OpenAICompatibleProvider":
        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be set in .env to use the OpenRouter provider."
            )
        import openai

        client = openai.OpenAI(
            api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url
        )
        return cls(
            client,
            settings.openrouter_model,
            settings.llm_max_tokens,
            settings.llm_temperature,
            extra_headers={"X-Title": "FilingsRAG"},  # OpenRouter's optional attribution header
        )

    def complete(self, system: str, user: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            extra_headers=self._extra_headers,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost_usd(self.model, input_tokens, output_tokens),
        )


def get_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "claude":
        return ClaudeProvider.from_settings(settings)
    if settings.llm_provider == "openai":
        return OpenAICompatibleProvider.from_settings_openai(settings)
    return OpenAICompatibleProvider.from_settings_openrouter(settings)
