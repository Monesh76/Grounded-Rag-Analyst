"""Tests for pipeline.ask(). Fully offline: `llm` is a fake, `retrieve` is a
canned function, so no DB, embedder, reranker or real LLM is ever touched.
"""

from filings_rag.config import get_settings
from filings_rag.generate.models import LLMResponse
from filings_rag.pipeline import ask, ask_detailed
from filings_rag.retrieve.models import RetrievalResult


def chunk(id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        id=id,
        doc_id="doc",
        ticker="AAPL",
        company="Apple",
        fiscal_year=2024,
        item="7",
        section_title="MD&A",
        page=23,
        text="Net revenue was $391 billion.",
        score=score,
    )


class FakeLLMProvider:
    model = "fake-model"

    def __init__(
        self, text: str, input_tokens: int = 10, output_tokens: int = 5, cost: float = 0.001
    ):
        self._text = text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cost = cost
        self.last_call: tuple[str, str] | None = None

    def complete(self, system: str, user: str) -> LLMResponse:
        self.last_call = (system, user)
        return LLMResponse(
            text=self._text,
            model=self.model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=self._cost,
        )


def settings_with(**overrides):
    return get_settings().model_copy(update={"evidence_gate_threshold": 0.5, **overrides})


def test_grounded_answer_with_valid_citation() -> None:
    results = [chunk("AAPL_2024_7_0")]
    llm = FakeLLMProvider("Net revenue was $391 billion [c:AAPL_2024_7_0].")

    answer = ask("What was revenue?", settings_with(), llm=llm, retrieve=lambda q: results)

    assert answer.grounded is True
    assert "[c:AAPL_2024_7_0]" in answer.answer
    assert [s.id for s in answer.sources] == ["AAPL_2024_7_0"]
    assert answer.tokens == 15
    assert answer.cost_usd == 0.001
    assert answer.latency_ms >= 0


def test_refuses_without_calling_llm_when_no_results() -> None:
    llm = FakeLLMProvider("should never be returned")
    answer = ask("anything", settings_with(), llm=llm, retrieve=lambda q: [])

    assert answer.answer == "Not found in the filings."
    assert answer.grounded is False
    assert answer.sources == []
    assert answer.tokens == 0
    assert llm.last_call is None  # the LLM was never invoked


def test_refuses_without_calling_llm_when_top_score_below_threshold() -> None:
    llm = FakeLLMProvider("should never be returned")
    results = [chunk("x", score=0.1)]  # below the 0.5 threshold in settings_with()

    answer = ask("anything", settings_with(), llm=llm, retrieve=lambda q: results)

    assert answer.answer == "Not found in the filings."
    assert answer.grounded is False
    assert llm.last_call is None


def test_evidence_gate_uses_configured_threshold() -> None:
    llm = FakeLLMProvider("An answer [c:x].")
    results = [chunk("x", score=0.6)]  # above the 0.5 threshold

    answer = ask("anything", settings_with(), llm=llm, retrieve=lambda q: results)

    assert answer.answer != "Not found in the filings."
    assert llm.last_call is not None


def test_ungrounded_when_llm_cites_an_invalid_id() -> None:
    results = [chunk("real_id")]
    llm = FakeLLMProvider("An answer [c:fabricated_id].")

    answer = ask("q", settings_with(), llm=llm, retrieve=lambda q: results)

    assert answer.grounded is False
    assert "[c:fabricated_id]" not in answer.answer  # stripped, not shown as a real source
    assert answer.sources == []


def test_ungrounded_when_llm_gives_no_citations_at_all() -> None:
    results = [chunk("real_id")]
    llm = FakeLLMProvider("An answer with no citation at all.")

    answer = ask("q", settings_with(), llm=llm, retrieve=lambda q: results)

    assert answer.grounded is False
    assert answer.sources == []


def test_ungrounded_when_llm_itself_refuses_despite_passing_evidence_gate() -> None:
    results = [chunk("real_id")]
    llm = FakeLLMProvider("Not found in the filings.")

    answer = ask("q", settings_with(), llm=llm, retrieve=lambda q: results)

    assert answer.answer == "Not found in the filings."
    assert answer.grounded is False


def test_refusal_is_truncated_even_when_the_model_pads_it() -> None:
    # Real behavior observed running this against a live model: told to reply
    # with "exactly this sentence and nothing else", it still added an
    # explanatory paragraph after judging the evidence insufficient.
    results = [chunk("real_id")]
    llm = FakeLLMProvider(
        "Not found in the filings.\n\nThe excerpts don't mention this specific figure, "
        "though they do discuss related topics."
    )

    answer = ask("q", settings_with(), llm=llm, retrieve=lambda q: results)

    assert answer.answer == "Not found in the filings."
    assert answer.grounded is False
    assert answer.tokens == 15  # still reflects the real LLM call that was made
    assert answer.cost_usd == 0.001


def test_passes_question_and_chunk_context_to_the_llm() -> None:
    results = [chunk("AAPL_2024_7_0")]
    llm = FakeLLMProvider("An answer [c:AAPL_2024_7_0].")

    ask("What was Apple's revenue?", settings_with(), llm=llm, retrieve=lambda q: results)

    system, user = llm.last_call
    assert "Not found in the filings." in system
    assert "What was Apple's revenue?" in user
    assert "[c:AAPL_2024_7_0]" in user


def test_ask_detailed_exposes_retrieved_chunks_and_raw_llm_text() -> None:
    results = [chunk("AAPL_2024_7_0")]
    llm = FakeLLMProvider("An answer [c:AAPL_2024_7_0] and [c:fabricated].")

    result = ask_detailed("q", settings_with(), llm=llm, retrieve=lambda q: results)

    assert result.retrieved == results
    assert result.raw_answer_text == "An answer [c:AAPL_2024_7_0] and [c:fabricated]."
    # The final Answer is still validated (fabricated citation stripped):
    assert "[c:fabricated]" not in result.answer.answer
    assert result.answer.grounded is False


def test_ask_detailed_has_no_retrieved_or_raw_text_when_evidence_gate_refuses() -> None:
    llm = FakeLLMProvider("should never be called")

    result = ask_detailed("q", settings_with(), llm=llm, retrieve=lambda q: [])

    assert result.retrieved == []
    assert result.raw_answer_text is None
    assert result.answer.answer == "Not found in the filings."


def test_ask_detailed_exposes_retrieved_chunks_even_when_llm_itself_refuses() -> None:
    results = [chunk("real_id")]
    llm = FakeLLMProvider("Not found in the filings.")

    result = ask_detailed("q", settings_with(), llm=llm, retrieve=lambda q: results)

    # Retrieval still happened and is exposed, even though the model refused --
    # useful for the eval harness to still measure Recall@k/MRR on this row.
    assert result.retrieved == results
    assert result.raw_answer_text == "Not found in the filings."
    assert result.answer.answer == "Not found in the filings."
