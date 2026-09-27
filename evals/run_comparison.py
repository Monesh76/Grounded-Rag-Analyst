"""python -m evals.run_comparison [--no-judge] [--only a,b,c]

Runs PLAN.md's P6 experiment grid (configs A-E) through the full pipeline and
writes results/comparison.md, one row per run. Each run also writes its own
results/<run-id>-full.json/.md via the same path `make eval-full` uses, so a
single run's detail is always inspectable, not just the summary row.

Costs money by default: DeepEval's FaithfulnessMetric makes several LLM calls
per question internally, at whichever model build_judge() picks (Claude Sonnet
5 here) -- a real run of all five configs cost $15+ once discovered, far more
than a first estimate assumed. --no-judge skips all of that (faithfulness/
correctness read "n/a" in comparison.md); --only reruns just the given configs
and reuses the others' already-written results/<name>-full.json rather than
re-running (and re-paying for) runs that already have real numbers.
"""

import argparse
import json
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


def run_grid(use_judge: bool = True, names: list[str] | None = None) -> list[dict]:
    rows = load_golden()
    summaries = []
    for name in names or GRID:
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
    # cost_per_1k_usd includes judge cost when the run was judged and doesn't
    # when it wasn't -- comparing it directly across a mix of judged/unjudged
    # runs would make hybrid_rerank look 15-100x cheaper than dense retrieval
    # for no reason connected to retrieval quality. Generation cost alone is
    # always apples-to-apples, so that's what the table reports; a "Judged"
    # column says which runs also paid for faithfulness/correctness.
    lines = [
        "# Experiment grid comparison (PLAN.md P6)",
        "",
        "Cost is generation-only and comparable across all rows. Faithfulness/"
        "correctness ('n/a' below) need a separate judge pass -- see 'Judged'.",
        "",
        "| Run | Chunking | Retrieval | Model | Judged | Recall@6 | MRR | Faithfulness | "
        "Correctness | Refusal accuracy | p95 latency (ms) | Generation cost/1K |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        judged = s.get("judge_cost_usd", 0.0) > 0
        faithfulness = f"{s['faithfulness']:.3f}" if "faithfulness" in s else "n/a"
        correctness = f"{s['correctness']:.3f}" if "correctness" in s else "n/a"
        generation_cost_per_1k = s["generation_cost_usd"] / s["num_questions"] * 1000
        lines.append(
            f"| {s['run_name'].upper()} | {s['chunking']} | {s['mode']} | {s['model']} | "
            f"{'yes' if judged else 'no'} | {s['recall_at_6']:.3f} | {s['mrr']:.3f} | "
            f"{faithfulness} | {correctness} | {s['refusal_accuracy']:.3f} | "
            f"{s['p95_latency_ms']:.0f} | ${generation_cost_per_1k:.2f} |"
        )
    path = Path("results/comparison.md")
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated config names to (re)run, e.g. 'd,e'. Others are "
        "read from their existing results/<name>-full.json instead of re-running "
        "(and re-paying for) them.",
    )
    args = parser.parse_args()

    for name in GRID:
        config = load_experiment_config(str(CONFIGS_DIR / f"{name}.yaml"))
        if config["mode"] not in MODES:
            raise ValueError(f"{name}.yaml: unknown mode {config['mode']!r}")

    only = args.only.split(",") if args.only else GRID
    unknown = set(only) - set(GRID)
    if unknown:
        raise ValueError(f"--only has unknown config name(s) {sorted(unknown)}; choose from {GRID}")

    fresh = {s["run_name"]: s for s in run_grid(use_judge=not args.no_judge, names=only)}
    summaries = []
    for name in GRID:
        if name in fresh:
            summaries.append(fresh[name])
            continue
        existing = Path("results") / f"{name}-full.json"
        if not existing.exists():
            raise FileNotFoundError(
                f"{existing} doesn't exist yet -- run without --only first, or include "
                f"{name!r} in --only."
            )
        summaries.append(json.loads(existing.read_text()))

    path = write_comparison(summaries)
    print(f"Wrote {path}")
    print(path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
