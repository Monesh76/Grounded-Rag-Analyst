"""Offline tests for run_comparison.py's pure helpers. The 5x-eval-full
orchestration itself is exercised for real via `make comparison`, the same
pattern as run_evals.py's own orchestration."""

from evals.run_comparison import CONFIGS_DIR, GRID, _model_label, write_comparison
from evals.run_evals import load_experiment_config
from filings_rag.config import get_settings
from filings_rag.retrieve.search import MODES


def test_all_five_grid_configs_exist_and_are_valid() -> None:
    for name in GRID:
        config = load_experiment_config(str(CONFIGS_DIR / f"{name}.yaml"))
        assert config["mode"] in MODES
        assert config["table"] in ("chunks", "chunks_fixed512")


def test_model_label_reflects_provider() -> None:
    settings = get_settings().model_copy(update={"llm_provider": "claude"})
    assert _model_label(settings) == settings.anthropic_model

    settings = get_settings().model_copy(update={"llm_provider": "openrouter"})
    assert _model_label(settings) == settings.openrouter_model


def test_write_comparison_includes_all_runs(tmp_path, monkeypatch) -> None:
    import evals.run_comparison as run_comparison_module

    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    summaries = [
        {
            "run_name": "a",
            "chunking": "fixed-512",
            "mode": "dense",
            "model": "anthropic/claude-haiku-4.5",
            "recall_at_6": 0.8,
            "mrr": 0.7,
            "refusal_accuracy": 0.9,
            "p95_latency_ms": 500.0,
            "cost_per_1k_usd": 1.0,
        },
        {
            "run_name": "e",
            "chunking": "section-aware",
            "mode": "hybrid_rerank",
            "model": "openai/gpt-4o-mini",
            "recall_at_6": 0.9,
            "mrr": 0.8,
            "faithfulness": 0.95,
            "correctness": 0.9,
            "refusal_accuracy": 0.92,
            "p95_latency_ms": 700.0,
            "cost_per_1k_usd": 2.0,
        },
    ]
    path = write_comparison(summaries)
    text = path.read_text()
    assert "| A |" in text
    assert "| E |" in text
    assert "n/a" in text  # run A wasn't judged (no faithfulness/correctness)
    assert "0.950" in text
    assert run_comparison_module.Path("results/comparison.md").exists()
