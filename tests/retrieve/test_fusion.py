from filings_rag.retrieve.fusion import fuse_results, reciprocal_rank_fusion
from filings_rag.retrieve.models import RetrievalResult


def result(id: str) -> RetrievalResult:
    return RetrievalResult(
        id=id,
        doc_id="doc",
        ticker="AAPL",
        company="Apple",
        fiscal_year=2024,
        item="1",
        section_title="Business",
        page=1,
        text=f"text for {id}",
        score=0.0,
    )


# --- reciprocal_rank_fusion (hand-made rankings, per PLAN.md) ---


def test_top_of_a_single_ranking_wins() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert fused == ["a", "b", "c"]


def test_appearing_in_both_rankings_beats_appearing_in_only_one() -> None:
    # "b" is 2nd in both lists; "a" is 1st in one list but absent from the other.
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
    assert fused[0] == "b"


def test_first_place_in_both_rankings_wins_outright() -> None:
    fused = reciprocal_rank_fusion([["x", "y"], ["x", "z"]])
    assert fused[0] == "x"


def test_empty_ranking_is_ignored() -> None:
    assert reciprocal_rank_fusion([["a", "b"], []]) == ["a", "b"]


# --- fuse_results (maps fused ids back to full objects) ---


def test_fuse_results_returns_full_objects_in_fused_order() -> None:
    dense = [result("a"), result("b")]
    keyword = [result("c"), result("b")]

    fused = fuse_results([dense, keyword])

    assert [r.id for r in fused] == ["b", "a", "c"]
    assert all(isinstance(r, RetrievalResult) for r in fused)


def test_fuse_results_deduplicates_by_id() -> None:
    dense = [result("a")]
    keyword = [result("a")]
    fused = fuse_results([dense, keyword])
    assert len(fused) == 1
