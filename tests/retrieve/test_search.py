"""Integration tests for search()'s mode dispatch: fuses/truncates/reranks
correctly for each mode, against real Postgres with controlled rows.
"""

import psycopg
import pytest

from filings_rag.config import get_settings
from filings_rag.retrieve.models import Filters
from filings_rag.retrieve.rerank import Reranker
from filings_rag.retrieve.search import search
from tests.retrieve.conftest import insert_chunk, unit_vector

pytestmark = pytest.mark.integration


class FixedEmbedder:
    dimensions = 1024

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


class KeywordCountReranker:
    """Fake cross-encoder: scores by how often "widget" appears in the text --
    deterministic, and distinguishable from the raw retrieval scores."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(text.lower().count("widget")) for _, text in pairs]


@pytest.fixture
def settings():
    s = get_settings().model_copy(
        update={
            "retrieval_dense_top_k": 10,
            "retrieval_keyword_top_k": 10,
            "retrieval_rerank_top_k": 3,
        }
    )
    return s


def test_dense_mode_returns_vector_ranked_results(conn: psycopg.Connection, settings) -> None:
    insert_chunk(conn, "close", "widget text", unit_vector(0))
    insert_chunk(conn, "far", "unrelated text", unit_vector(1))

    results = search(conn, FixedEmbedder(unit_vector(0)), None, "widget", "dense", settings)

    assert [r.id for r in results] == ["close", "far"]


def test_keyword_mode_returns_text_ranked_results(conn: psycopg.Connection, settings) -> None:
    insert_chunk(conn, "match", "widget widget widget", unit_vector(0))
    insert_chunk(conn, "nomatch", "unrelated content here", unit_vector(1))

    results = search(conn, FixedEmbedder(unit_vector(0)), None, "widget", "keyword", settings)

    assert [r.id for r in results] == ["match"]


def test_hybrid_mode_fuses_dense_and_keyword(conn: psycopg.Connection, settings) -> None:
    # "dense_only" ranks #1 by vector similarity but doesn't match the keyword search
    # at all (0 lexeme overlap -- excluded from that ranking entirely, not just
    # ranked low in it). "both" ranks a modest 2nd in *each* signal. Appearing in
    # both rankings should beat being #1 in only one -- the actual point of fusion.
    insert_chunk(conn, "dense_only", "irrelevant text with no keyword", unit_vector(0))
    insert_chunk(conn, "keyword_only", "widget widget widget widget", unit_vector(5))
    insert_chunk(conn, "both", "widget mentioned once", unit_vector(1))

    results = search(conn, FixedEmbedder(unit_vector(0)), None, "widget", "hybrid", settings)

    assert [r.id for r in results].index("both") < [r.id for r in results].index("dense_only")


def test_hybrid_rerank_mode_applies_the_reranker(conn: psycopg.Connection, settings) -> None:
    insert_chunk(conn, "no_widget", "widget", unit_vector(0))
    insert_chunk(conn, "many_widget", "widget widget widget widget widget", unit_vector(1))

    reranker = Reranker(KeywordCountReranker())
    results = search(
        conn, FixedEmbedder(unit_vector(0)), reranker, "widget", "hybrid_rerank", settings
    )

    assert results[0].id == "many_widget"  # reranker's own scoring wins, not vector distance


def test_hybrid_rerank_without_a_reranker_raises(conn: psycopg.Connection, settings) -> None:
    with pytest.raises(ValueError, match="requires a reranker"):
        search(conn, FixedEmbedder(unit_vector(0)), None, "widget", "hybrid_rerank", settings)


def test_unknown_mode_raises(conn: psycopg.Connection, settings) -> None:
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        search(conn, FixedEmbedder(unit_vector(0)), None, "widget", "bogus", settings)


def test_respects_explicit_top_k_override(conn: psycopg.Connection, settings) -> None:
    for i in range(5):
        insert_chunk(conn, f"c{i}", "widget", unit_vector(i))

    results = search(
        conn, FixedEmbedder(unit_vector(0)), None, "widget", "dense", settings, top_k=1
    )
    assert len(results) == 1


def test_explicit_filters_override_auto_extracted_ones(conn: psycopg.Connection, settings) -> None:
    # The question text mentions no ticker at all -- extract_filters(question)
    # would find nothing. An explicit Filters (e.g. from the UI's dropdowns)
    # must still apply.
    insert_chunk(conn, "match", "widget", unit_vector(0), ticker="TEST")
    insert_chunk(conn, "other", "widget", unit_vector(0), ticker="OTHER")

    results = search(
        conn,
        FixedEmbedder(unit_vector(0)),
        None,
        "widget",
        "dense",
        settings,
        filters=Filters(ticker="TEST"),
    )
    assert [r.id for r in results] == ["match"]
