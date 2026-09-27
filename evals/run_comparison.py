"""python -m evals.run_comparison [--no-judge]

Runs PLAN.md's P6 experiment grid (configs A-E) through the full pipeline and
writes results/comparison.md, one row per run. Each run also writes its own
results/<run-id>-full.json/.md via the same path `make eval-full` uses, so a
single run's detail is always inspectable, not just the summary row.

Costs money (5x eval-full, judge included by default) -- run deliberately,
not from CI.
"""

import argparse
import sys
from pathlib import Path

from evals.run_evals import load_experiment_config, load_golden, run_full, write_results
from filings_rag import tracing
from filings_rag.config import get_settings
from filings_rag.retrieve.search import MODES

CONFIGS_DIR = Path("experiments/configs")
GRID = ["a", "b", "c", "d", "e"]

# Which chunking strategy each run's `table` corresponds to, purely for the
# comparison table's "Chunking" column -- table names alone don't say this.
_CHUNKING_LABEL = {"chunks": "section-aware", "chunks_fixed512": "fixed-512"}


def run_grid(use_judge: bool = True) -> list[dict]:
    rows = load_golden()
    summaries = []
    for name in GRID:
        config = load_experiment_config(str(CONFIGS_DIR / f"{name}.yaml"))
        settings = get_settings().model_copy(update=config["settings"])
        summary = run_full(
            rows, settings, mode=config["mode"], use_judge=use_judge, table=config["table"]
        )
        summary["run_name"] = name
        summary["chunking"] = _CHUNKING_LABEL.get(config["table"], config["table"])
        summary["model"] = _model_label(settings)
        write_results(summary, f"{name}-full")
        tracing.flush(settings)
        summaries.append(summary)
    return summaries


def _model_label(settings) -> str:
    if settings.llm_provider == "claude":
        return settings.anthropic_model
    if settings.llm_provider == "openai":
        return settings.openai_model
    return settings.openrouter_model


def write_comparison(summaries: list[dict]) -> Path:
    lines = [
        "# Experiment grid comparison (PLAN.md P6)",
        "",
        "| Run | Chunking | Retrieval | Model | Recall@6 | MRR | Faithfulness | "
        "Correctness | Refusal accuracy | p95 latency (ms) | Cost/1K questions |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        faithfulness = f"{s['faithfulness']:.3f}" if "faithfulness" in s else "n/a"
        correctness = f"{s['correctness']:.3f}" if "correctness" in s else "n/a"
        lines.append(
            f"| {s['run_name'].upper()} | {s['chunking']} | {s['mode']} | {s['model']} | "
            f"{s['recall_at_6']:.3f} | {s['mrr']:.3f} | {faithfulness} | {correctness} | "
            f"{s['refusal_accuracy']:.3f} | {s['p95_latency_ms']:.0f} | "
            f"${s['cost_per_1k_usd']:.2f} |"
        )
    path = Path("results/comparison.md")
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    assert set(GRID) <= {"a", "b", "c", "d", "e"}
    for name in GRID:
        config = load_experiment_config(str(CONFIGS_DIR / f"{name}.yaml"))
        if config["mode"] not in MODES:
            raise ValueError(f"{name}.yaml: unknown mode {config['mode']!r}")

    summaries = run_grid(use_judge=not args.no_judge)
    path = write_comparison(summaries)
    print(f"Wrote {path}")
    print(path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
