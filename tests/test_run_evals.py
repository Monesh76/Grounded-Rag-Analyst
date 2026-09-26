"""Offline tests for run_evals.py's pure helpers (loading, formatting). The
orchestration functions (run_retrieval_only/run_full) are exercised for real
via `make eval`/`make eval-full`, the same pattern as this project's other CLI
entry points.
"""

import json
from pathlib import Path

import pytest

from evals.run_evals import (
    GoldenRow,
    _mean,
    _to_markdown,
    _to_source_tuples,
    load_experiment_config,
    load_golden,
    write_results,
)
from filings_rag.retrieve.models import RetrievalResult


def result(ticker: str, fiscal_year: int, item: str) -> RetrievalResult:
    return RetrievalResult(
        id=f"{ticker}_{fiscal_year}_{item}_0",
        doc_id="doc",
        ticker=ticker,
        company=ticker,
        fiscal_year=fiscal_year,
        item=item,
        section_title="Section",
        page=1,
        text="text",
        score=0.9,
    )


# --- load_golden ---


def test_load_golden_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q001",
                "question": "What was revenue?",
                "type": "single_fact",
                "expected_answer": "$1",
                "expected_sources": [{"doc": "AAPL_2025", "section": "7"}],
                "must_refuse": False,
                "verified": True,
            }
        )
        + "\n"
    )
    rows = load_golden(path)
    assert len(rows) == 1
    assert isinstance(rows[0], GoldenRow)
    assert rows[0].id == "q001"


def test_load_golden_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    row = {
        "id": "q001",
        "question": "q",
        "type": "unanswerable",
        "expected_answer": "Not found in the filings.",
        "expected_sources": [],
        "must_refuse": True,
        "verified": True,
    }
    path.write_text(json.dumps(row) + "\n\n" + json.dumps(row) + "\n")
    assert len(load_golden(path)) == 2


# --- load_experiment_config ---


def test_load_experiment_config_defaults_when_no_path_given() -> None:
    config = load_experiment_config(None)
    assert config["name"] == "default"
    assert config["mode"] == "hybrid_rerank"
    assert config["settings"] == {}


def test_load_experiment_config_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text("name: my_experiment\nmode: dense\n")
    config = load_experiment_config(str(path))
    assert config["name"] == "my_experiment"
    assert config["mode"] == "dense"
    assert config["settings"] == {}  # defaulted, since not present in the file


# --- _to_source_tuples ---


def test_to_source_tuples_maps_ticker_fiscal_year_and_item() -> None:
    results = [result("AAPL", 2025, "7"), result("MSFT", 2024, "8")]
    assert _to_source_tuples(results) == [("AAPL_2025", "7"), ("MSFT_2024", "8")]


# --- _mean ---


def test_mean_of_values() -> None:
    assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_of_empty_is_zero() -> None:
    assert _mean([]) == 0.0


# --- write_results / _to_markdown ---


def test_write_results_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evals.run_evals as run_evals_module

    monkeypatch.setattr(run_evals_module, "RESULTS_DIR", tmp_path)
    summary = {
        "kind": "retrieval",
        "mode": "hybrid_rerank",
        "num_questions": 10,
        "recall_at_6": 0.9,
        "mrr": 0.8,
        "per_row": [],
    }
    json_path, md_path = write_results(summary, "test-run")

    assert json_path == tmp_path / "test-run.json"
    assert json.loads(json_path.read_text())["recall_at_6"] == 0.9
    assert "Recall@6" in md_path.read_text()
    assert "0.900" in md_path.read_text()


def test_markdown_includes_full_mode_metrics() -> None:
    summary = {
        "kind": "full",
        "mode": "hybrid_rerank",
        "num_questions": 5,
        "recall_at_6": 1.0,
        "mrr": 1.0,
        "citation_precision": 1.0,
        "refusal_accuracy": 0.9,
        "p50_latency_ms": 500.0,
        "p95_latency_ms": 900.0,
        "cost_per_1k_usd": 12.5,
        "total_cost_usd": 0.0625,
        "cache_hits": 2,
        "cache_misses": 3,
        "faithfulness": 0.95,
        "correctness": 0.88,
    }
    md = _to_markdown(summary, "run-1")
    assert "Faithfulness" in md
    assert "Correctness" in md
    assert "Cost per 1K questions" in md
    assert "$12.50" in md
