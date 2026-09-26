"""Runs the golden set through retrieval only (`make eval`) or the full
pipeline (`make eval-full`), writing results/<run-id>.json and a matching
markdown summary.

`make eval` computes only Recall@6 and MRR: no LLM call, so it's free and safe
to run on every PR. `make eval-full` runs the real pipeline (LLM calls cached
by hash) and additionally computes citation precision, refusal accuracy,
latency percentiles, cost per 1K, and (unless --no-judge) DeepEval
faithfulness/correctness -- that's the one that costs money.
"""

import argparse
import statistics
import sys
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel

from evals.llm_cache import CachingLLMProvider
from evals.metrics import (
    ExpectedSource,
    citation_precision,
    cost_per_1k_questions,
    mean_reciprocal_rank,
    percentile,
    recall_at_k,
    refusal_accuracy,
)
from filings_rag.config import Settings, get_settings
from filings_rag.db import get_connection
from filings_rag.embed import get_embedder
from filings_rag.generate.citations import find_citation_ids
from filings_rag.generate.llm import get_llm_provider
from filings_rag.pipeline import ask_detailed
from filings_rag.retrieve.models import RetrievalResult
from filings_rag.retrieve.rerank import Reranker
from filings_rag.retrieve.search import search

# The embedder and reranker are shared across worker threads below (loading
# them once, not per-question, is what actually fixes P4's flagged latency
# issue). Capping torch to one thread per call avoids each inference call
# spawning its own multi-threaded BLAS kernel -- but that alone isn't enough:
# confirmed by hand that calling the shared CrossEncoder's .predict() from
# multiple threads concurrently causes severe thrashing (and, once, an
# outright segfault) regardless of this setting. _MODEL_LOCK below is what
# actually fixes it: local-model inference is serialized across workers, while
# each worker's LLM call (the real bottleneck in eval-full, seconds of network
# I/O) still runs concurrently, outside the lock.
torch.set_num_threads(1)
_MODEL_LOCK = threading.Lock()

GOLDEN_PATH = Path("evals/golden.jsonl")
RESULTS_DIR = Path("results")
# Retrieval-only is entirely CPU-bound (embedding + reranking, no network) --
# concurrency doesn't overlap anything here, it only adds thread-scheduling
# overhead, so this stays sequential by default.
DEFAULT_RETRIEVAL_CONCURRENCY = 1
# Full pipeline is dominated by network-bound LLM calls (seconds each), where
# concurrency genuinely overlaps waiting time across questions; kept modest
# since each worker still does its own CPU-bound reranking too.
DEFAULT_FULL_CONCURRENCY = 3


class GoldenRow(BaseModel):
    id: str
    question: str
    type: str
    expected_answer: str
    expected_sources: list[dict[str, str]]
    must_refuse: bool
    verified: bool


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenRow]:
    lines = path.read_text().splitlines()
    return [GoldenRow.model_validate_json(line) for line in lines if line.strip()]


def load_experiment_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {"name": "default", "mode": "hybrid_rerank", "settings": {}}
    config = yaml.safe_load(Path(path).read_text()) or {}
    config.setdefault("mode", "hybrid_rerank")
    config.setdefault("settings", {})
    return config


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def _to_source_tuples(results: list[RetrievalResult]) -> list[tuple[str, str]]:
    return [(f"{r.ticker}_{r.fiscal_year}", r.item) for r in results]


def _run_concurrent(
    fn: Callable[[GoldenRow], dict[str, Any]], rows: list[GoldenRow], concurrency: int
) -> list[dict[str, Any]]:
    if concurrency <= 1:
        return [fn(row) for row in rows]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(fn, rows))


# --- make eval: retrieval only, no LLM call ---


def run_retrieval_only(
    rows: list[GoldenRow],
    settings: Settings,
    mode: str = "hybrid_rerank",
    concurrency: int = DEFAULT_RETRIEVAL_CONCURRENCY,
) -> dict[str, Any]:
    embedder = get_embedder(settings)
    reranker = Reranker.from_settings(settings) if mode == "hybrid_rerank" else None
    answerable = [r for r in rows if r.expected_sources]

    def process(row: GoldenRow) -> dict[str, Any]:
        conn = get_connection(settings)
        try:
            with _MODEL_LOCK:
                results = search(conn, embedder, reranker, row.question, mode, settings)
        finally:
            conn.close()
        retrieved = _to_source_tuples(results)
        expected = [ExpectedSource(**s) for s in row.expected_sources]
        return {
            "id": row.id,
            "recall": recall_at_k(retrieved, expected),
            "mrr": mean_reciprocal_rank(retrieved, expected),
        }

    per_row = _run_concurrent(process, answerable, concurrency)
    return {
        "kind": "retrieval",
        "mode": mode,
        "num_questions": len(answerable),
        "recall_at_6": _mean(r["recall"] for r in per_row),
        "mrr": _mean(r["mrr"] for r in per_row),
        "per_row": per_row,
    }


# --- make eval-full: the real pipeline, LLM calls cached by hash ---


def run_full(
    rows: list[GoldenRow],
    settings: Settings,
    mode: str = "hybrid_rerank",
    concurrency: int = DEFAULT_FULL_CONCURRENCY,
    use_judge: bool = True,
) -> dict[str, Any]:
    embedder = get_embedder(settings)
    reranker = Reranker.from_settings(settings)
    llm = CachingLLMProvider(get_llm_provider(settings))
    judge = None
    if use_judge:
        from evals.judge import build_judge, score_answer

        judge = build_judge(settings)

    def locked_search(conn: Any, question: str) -> list[RetrievalResult]:
        # Serializes the CPU-bound embed+rerank step across workers; the LLM
        # call in ask_detailed() below happens outside this function, so it
        # still runs concurrently across workers.
        with _MODEL_LOCK:
            return search(conn, embedder, reranker, question, mode, settings)

    def process(row: GoldenRow) -> dict[str, Any]:
        # One question's API error (rate limit, transient outage, ...) must not
        # abort the whole run and lose every other question's results -- real
        # failure mode hit while building this: OpenRouter's in-flight budget
        # limit killed a run partway through with nothing written to disk.
        try:
            return _process_row(row)
        except Exception as exc:  # noqa: BLE001
            return {"id": row.id, "type": row.type, "error": str(exc)}

    def _process_row(row: GoldenRow) -> dict[str, Any]:
        conn = get_connection(settings)
        try:
            result = ask_detailed(
                row.question, settings, llm=llm, retrieve=lambda q: locked_search(conn, q)
            )
        finally:
            conn.close()

        answer = result.answer
        retrieved_tuples = _to_source_tuples(result.retrieved)
        retrieved_ids = {r.id for r in result.retrieved}
        raw_cited_ids = find_citation_ids(result.raw_answer_text) if result.raw_answer_text else []
        expected = [ExpectedSource(**s) for s in row.expected_sources]
        actually_refused = answer.answer.strip() == settings.refusal_text

        row_result: dict[str, Any] = {
            "id": row.id,
            "type": row.type,
            "recall": recall_at_k(retrieved_tuples, expected) if expected else None,
            "mrr": mean_reciprocal_rank(retrieved_tuples, expected) if expected else None,
            "citation_precision": citation_precision(raw_cited_ids, retrieved_ids),
            "must_refuse": row.must_refuse,
            "actually_refused": actually_refused,
            "refusal_correct": actually_refused == row.must_refuse,
            "grounded": answer.grounded,
            "latency_ms": answer.latency_ms,
            "tokens": answer.tokens,
            "cost_usd": answer.cost_usd,
            "answer": answer.answer,
        }

        # Faithfulness/correctness are Generation-stage metrics: skip them for
        # genuinely unanswerable rows (already covered by refusal_accuracy) to
        # avoid paying the judge to score a trivial "Not found" vs "Not found".
        if judge is not None and not row.must_refuse:
            scores = score_answer(
                judge,
                row.question,
                answer.answer,
                row.expected_answer,
                [r.text for r in result.retrieved],
            )
            row_result["faithfulness"] = scores.faithfulness
            row_result["correctness"] = scores.correctness

        return row_result

    per_row = _run_concurrent(process, rows, concurrency)

    errored_rows = [r for r in per_row if "error" in r]
    ok_rows = [r for r in per_row if "error" not in r]
    answerable_rows = [r for r in ok_rows if r["recall"] is not None]
    judged_rows = [r for r in ok_rows if "faithfulness" in r]
    latencies = [r["latency_ms"] for r in ok_rows]
    total_cost = sum(r["cost_usd"] for r in ok_rows)

    summary = {
        "kind": "full",
        "mode": mode,
        "num_questions": len(rows),
        "num_errors": len(errored_rows),
        "recall_at_6": _mean(r["recall"] for r in answerable_rows),
        "mrr": _mean(r["mrr"] for r in answerable_rows),
        "citation_precision": _mean(r["citation_precision"] for r in ok_rows),
        "refusal_accuracy": refusal_accuracy(
            [(r["must_refuse"], r["actually_refused"]) for r in ok_rows]
        ),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "cost_per_1k_usd": cost_per_1k_questions(total_cost, len(rows)),
        "total_cost_usd": total_cost,
        "cache_hits": llm.hits,
        "cache_misses": llm.misses,
        "per_row": per_row,
    }
    if judged_rows:
        summary["faithfulness"] = _mean(r["faithfulness"] for r in judged_rows)
        summary["correctness"] = _mean(r["correctness"] for r in judged_rows)
    return summary


# --- output ---


def write_results(summary: dict[str, Any], run_id: str) -> tuple[Path, Path]:
    import json

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{run_id}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    md_path = RESULTS_DIR / f"{run_id}.md"
    md_path.write_text(_to_markdown(summary, run_id))
    return json_path, md_path


def _to_markdown(summary: dict[str, Any], run_id: str) -> str:
    header = (
        f"Mode: `{summary['mode']}` | Kind: `{summary['kind']}` | "
        f"Questions: {summary['num_questions']}"
    )
    lines = [f"# Eval results: {run_id}", "", header, ""]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Recall@6 | {summary['recall_at_6']:.3f} |")
    lines.append(f"| MRR | {summary['mrr']:.3f} |")
    if summary["kind"] == "full":
        if summary.get("num_errors"):
            lines.append(f"| Errors | {summary['num_errors']} of {summary['num_questions']} |")
        lines.append(f"| Citation precision | {summary['citation_precision']:.3f} |")
        lines.append(f"| Refusal accuracy | {summary['refusal_accuracy']:.3f} |")
        if "faithfulness" in summary:
            lines.append(f"| Faithfulness | {summary['faithfulness']:.3f} |")
            lines.append(f"| Correctness | {summary['correctness']:.3f} |")
        lines.append(f"| p50 latency (ms) | {summary['p50_latency_ms']:.0f} |")
        lines.append(f"| p95 latency (ms) | {summary['p95_latency_ms']:.0f} |")
        lines.append(f"| Cost per 1K questions | ${summary['cost_per_1k_usd']:.2f} |")
        lines.append(f"| Total cost (this run) | ${summary['total_cost_usd']:.4f} |")
        lines.append(
            f"| LLM cache hits/misses | {summary['cache_hits']}/{summary['cache_misses']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run the full pipeline (costs money)")
    parser.add_argument("--config", default=None, help="Path to an experiment YAML")
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip DeepEval faithfulness/correctness"
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    settings = get_settings().model_copy(update=config["settings"])
    rows = load_golden()
    kind = "full" if args.full else "retrieval"
    timestamp = f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
    run_id = args.run_id or f"{timestamp}-{config['name']}-{kind}"

    if args.full:
        summary = run_full(rows, settings, mode=config["mode"], use_judge=not args.no_judge)
    else:
        summary = run_retrieval_only(rows, settings, mode=config["mode"])

    json_path, md_path = write_results(summary, run_id)
    print(f"Wrote {json_path} and {md_path}")
    print(_to_markdown(summary, run_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
