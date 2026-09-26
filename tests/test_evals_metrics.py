import pytest

from evals.metrics import (
    ExpectedSource,
    citation_precision,
    cost_per_1k_questions,
    mean_reciprocal_rank,
    percentile,
    recall_at_k,
    refusal_accuracy,
)

# --- recall_at_k ---


def test_recall_full_when_all_expected_sources_found() -> None:
    retrieved = [("AAPL_2025", "7"), ("AAPL_2025", "1A"), ("MSFT_2025", "8")]
    expected = [ExpectedSource("AAPL_2025", "7")]
    assert recall_at_k(retrieved, expected) == 1.0


def test_recall_zero_when_expected_source_missing() -> None:
    retrieved = [("MSFT_2025", "8")]
    expected = [ExpectedSource("AAPL_2025", "7")]
    assert recall_at_k(retrieved, expected) == 0.0


def test_recall_is_partial_credit_for_multi_source_questions() -> None:
    retrieved = [("AAPL_2025", "7")]  # only one of two expected sources present
    expected = [ExpectedSource("AAPL_2025", "7"), ExpectedSource("MSFT_2025", "8")]
    assert recall_at_k(retrieved, expected) == 0.5


def test_recall_vacuously_true_for_no_expected_sources() -> None:
    assert recall_at_k([("AAPL_2025", "7")], []) == 1.0


# --- mean_reciprocal_rank ---


def test_mrr_is_one_when_expected_source_ranks_first() -> None:
    retrieved = [("AAPL_2025", "7"), ("MSFT_2025", "8")]
    expected = [ExpectedSource("AAPL_2025", "7")]
    assert mean_reciprocal_rank(retrieved, expected) == 1.0


def test_mrr_reflects_rank_position() -> None:
    retrieved = [("MSFT_2025", "8"), ("GOOGL_2025", "8"), ("AAPL_2025", "7")]
    expected = [ExpectedSource("AAPL_2025", "7")]
    assert mean_reciprocal_rank(retrieved, expected) == pytest.approx(1 / 3)


def test_mrr_zero_when_not_found() -> None:
    retrieved = [("MSFT_2025", "8")]
    expected = [ExpectedSource("AAPL_2025", "7")]
    assert mean_reciprocal_rank(retrieved, expected) == 0.0


def test_mrr_uses_first_expected_source_found_not_best_rank_overall() -> None:
    # Two expected sources; only the second-ranked item matches one of them.
    retrieved = [("X", "1"), ("AAPL_2025", "7")]
    expected = [ExpectedSource("MSFT_2025", "8"), ExpectedSource("AAPL_2025", "7")]
    assert mean_reciprocal_rank(retrieved, expected) == pytest.approx(1 / 2)


# --- citation_precision ---


def test_citation_precision_full_when_all_cited_ids_were_retrieved() -> None:
    assert citation_precision(["a", "b"], {"a", "b", "c"}) == 1.0


def test_citation_precision_partial_when_some_cited_ids_are_fabricated() -> None:
    assert citation_precision(["a", "fake"], {"a", "b"}) == 0.5


def test_citation_precision_vacuous_when_no_citations() -> None:
    assert citation_precision([], {"a"}) == 1.0


# --- refusal_accuracy ---


def test_refusal_accuracy_full_when_all_match() -> None:
    results = [(True, True), (False, False), (True, True)]
    assert refusal_accuracy(results) == 1.0


def test_refusal_accuracy_penalizes_wrong_refusal() -> None:
    # One row should have refused but the pipeline answered instead.
    results = [(True, True), (True, False)]
    assert refusal_accuracy(results) == 0.5


def test_refusal_accuracy_penalizes_unwanted_refusal() -> None:
    # One row should have answered but the pipeline refused instead.
    results = [(False, False), (False, True)]
    assert refusal_accuracy(results) == 0.5


def test_refusal_accuracy_vacuous_for_no_rows() -> None:
    assert refusal_accuracy([]) == 1.0


# --- percentile ---


def test_percentile_50_is_median_for_odd_count() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3


def test_percentile_0_and_100_are_min_and_max() -> None:
    values = [5, 1, 3, 2, 4]
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 5


def test_percentile_95_of_uniform_latencies() -> None:
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 95) == pytest.approx(95.05, abs=0.5)


def test_percentile_empty_is_zero() -> None:
    assert percentile([], 50) == 0.0


def test_percentile_single_value() -> None:
    assert percentile([42.0], 95) == 42.0


# --- cost_per_1k_questions ---


def test_cost_per_1k_scales_linearly() -> None:
    assert cost_per_1k_questions(total_cost_usd=0.5, num_questions=50) == pytest.approx(10.0)


def test_cost_per_1k_zero_questions_is_zero() -> None:
    assert cost_per_1k_questions(total_cost_usd=1.0, num_questions=0) == 0.0
