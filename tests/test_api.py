"""Tests for the FastAPI app. Fully offline: get_ask_fn is overridden with a
fake, so no real DB, embedder, reranker or LLM is ever touched.
"""

from fastapi.testclient import TestClient

from filings_rag.api import app, get_ask_fn
from filings_rag.config import Settings
from filings_rag.generate.models import Answer, Source

client = TestClient(app)

GROUNDED_ANSWER = Answer(
    answer="Net revenue was $391 billion [c:AAPL_2024_7_0].",
    sources=[
        Source(
            id="AAPL_2024_7_0",
            ticker="AAPL",
            company="Apple",
            fiscal_year=2024,
            item="7",
            section_title="MD&A",
            page=23,
            text="Net revenue was $391 billion, up 3% year over year.",
        )
    ],
    grounded=True,
    latency_ms=123.4,
    tokens=42,
    cost_usd=0.001,
)

REFUSAL_ANSWER = Answer(
    answer="Not found in the filings.",
    sources=[],
    grounded=False,
    latency_ms=50.0,
    tokens=0,
    cost_usd=0.0,
)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_grounded_answer_with_sources() -> None:
    app.dependency_overrides[get_ask_fn] = lambda: lambda q, s, f=None: GROUNDED_ANSWER
    try:
        response = client.post("/ask", json={"question": "What was Apple's revenue?"})
    finally:
        app.dependency_overrides.pop(get_ask_fn, None)

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["sources"][0]["id"] == "AAPL_2024_7_0"
    assert "[c:AAPL_2024_7_0]" in body["answer"]
    assert body["tokens"] == 42


def test_ask_returns_refusal_for_unanswerable_question() -> None:
    app.dependency_overrides[get_ask_fn] = lambda: lambda q, s, f=None: REFUSAL_ANSWER
    try:
        response = client.post("/ask", json={"question": "irrelevant question"})
    finally:
        app.dependency_overrides.pop(get_ask_fn, None)

    body = response.json()
    assert body["answer"] == "Not found in the filings."
    assert body["grounded"] is False


def test_ask_passes_the_question_through() -> None:
    captured = {}

    def fake_ask(question: str, settings: Settings, filters=None) -> Answer:
        captured["question"] = question
        captured["filters"] = filters
        return REFUSAL_ANSWER

    app.dependency_overrides[get_ask_fn] = lambda: fake_ask
    try:
        client.post("/ask", json={"question": "What was Visa's net revenue?"})
    finally:
        app.dependency_overrides.pop(get_ask_fn, None)

    assert captured["question"] == "What was Visa's net revenue?"


def test_ask_passes_ticker_and_fiscal_year_as_filters() -> None:
    captured = {}

    def fake_ask(question: str, settings: Settings, filters=None) -> Answer:
        captured["filters"] = filters
        return REFUSAL_ANSWER

    app.dependency_overrides[get_ask_fn] = lambda: fake_ask
    try:
        client.post(
            "/ask", json={"question": "What was revenue?", "ticker": "AAPL", "fiscal_year": 2024}
        )
    finally:
        app.dependency_overrides.pop(get_ask_fn, None)

    assert captured["filters"].ticker == "AAPL"
    assert captured["filters"].fiscal_year == 2024


def test_ask_without_ticker_or_fiscal_year_passes_no_filters() -> None:
    captured = {}

    def fake_ask(question: str, settings: Settings, filters=None) -> Answer:
        captured["filters"] = filters
        return REFUSAL_ANSWER

    app.dependency_overrides[get_ask_fn] = lambda: fake_ask
    try:
        client.post("/ask", json={"question": "What was revenue?"})
    finally:
        app.dependency_overrides.pop(get_ask_fn, None)

    assert captured["filters"] is None


def test_ask_missing_question_is_a_422() -> None:
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_streaming_sends_text_then_a_final_json_line() -> None:
    app.dependency_overrides[get_ask_fn] = lambda: lambda q, s, f=None: GROUNDED_ANSWER
    try:
        response = client.post(
            "/ask", json={"question": "What was Apple's revenue?", "stream": True}
        )
    finally:
        app.dependency_overrides.pop(get_ask_fn, None)

    assert response.status_code == 200
    body = response.text
    assert "Net revenue was $391 billion" in body
    assert '"grounded":true' in body
    assert '"id":"AAPL_2024_7_0"' in body


def test_ask_streaming_never_sends_more_than_the_validated_text() -> None:
    # The streamed text portion (before the final JSON line) must reconstruct
    # exactly answer.answer -- nothing extra, nothing missing.
    app.dependency_overrides[get_ask_fn] = lambda: lambda q, s, f=None: GROUNDED_ANSWER
    try:
        response = client.post("/ask", json={"question": "q", "stream": True})
    finally:
        app.dependency_overrides.pop(get_ask_fn, None)

    streamed_text, _, _json_line = response.text.partition("\n\n")
    assert streamed_text == GROUNDED_ANSWER.answer
