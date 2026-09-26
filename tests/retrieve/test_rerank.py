from filings_rag.retrieve.models import RetrievalResult
from filings_rag.retrieve.rerank import Reranker


def result(id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        id=id,
        doc_id="doc",
        ticker="AAPL",
        company="Apple",
        fiscal_year=2024,
        item="1",
        section_title="Business",
        page=1,
        text=text,
        score=0.0,
    )


class FakeCrossEncoder:
    """Scores a pair by how many times the question's first word appears in the
    candidate text -- deterministic and easy to reason about in a test."""

    def __init__(self) -> None:
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.predict_calls.append(pairs)
        return [float(text.lower().count(question.split()[0].lower())) for question, text in pairs]


def test_reorders_by_relevance() -> None:
    candidates = [
        result("weak", "revenue grew"),
        result("strong", "revenue revenue revenue"),
    ]
    reranked = Reranker(FakeCrossEncoder()).rerank("revenue", candidates, top_k=2)
    assert [r.id for r in reranked] == ["strong", "weak"]


def test_respects_top_k() -> None:
    candidates = [result(str(i), "revenue " * i) for i in range(5)]
    reranked = Reranker(FakeCrossEncoder()).rerank("revenue", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].id == "4"  # most occurrences


def test_score_is_replaced_with_the_reranker_score() -> None:
    candidates = [result("a", "revenue revenue")]
    reranked = Reranker(FakeCrossEncoder()).rerank("revenue", candidates, top_k=1)
    assert reranked[0].score == 2.0


def test_empty_candidates_returns_empty_without_calling_predict() -> None:
    model = FakeCrossEncoder()
    result_list = Reranker(model).rerank("revenue", [], top_k=6)
    assert result_list == []
    assert model.predict_calls == []


def test_passes_question_paired_with_each_candidates_text() -> None:
    model = FakeCrossEncoder()
    candidates = [result("a", "text a"), result("b", "text b")]
    Reranker(model).rerank("my question", candidates, top_k=2)
    assert model.predict_calls == [[("my question", "text a"), ("my question", "text b")]]
