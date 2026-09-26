"""CI gates: fails the build if a golden-set metric is below its threshold.
Reads the most recently written results/*.json of the matching kind -- run
`make eval` or `make eval-full` first to generate one.

Per CLAUDE.md's eval rules: never lower a threshold to make a run pass --
report the regression and propose a fix instead.

Thresholds are PLAN.md section 5's starting points, "tighten later":
Recall@6 >= 0.80, faithfulness >= 0.85, refusal accuracy >= 0.90,
citation precision = 1.0.
"""

import json
from pathlib import Path

import pytest

RESULTS_DIR = Path("results")

RECALL_AT_6_THRESHOLD = 0.80
FAITHFULNESS_THRESHOLD = 0.85
REFUSAL_ACCURACY_THRESHOLD = 0.90
CITATION_PRECISION_THRESHOLD = 1.0


def _latest_result(kind: str) -> dict:
    matching = [
        path
        for path in sorted(RESULTS_DIR.glob("*.json"))
        if json.loads(path.read_text()).get("kind") == kind
    ]
    if not matching:
        pytest.fail(
            f"No '{kind}' eval results in results/ -- run "
            f"`make {'eval-full' if kind == 'full' else 'eval'}` first."
        )
    # filenames are timestamp-prefixed, so sorted() puts the newest one last.
    return json.loads(matching[-1].read_text())


def test_recall_at_6_meets_threshold() -> None:
    result = _latest_result("retrieval")
    assert result["recall_at_6"] >= RECALL_AT_6_THRESHOLD, (
        f"Recall@6 regressed to {result['recall_at_6']:.3f}, below the "
        f"{RECALL_AT_6_THRESHOLD} threshold. Do not lower this threshold to pass --"
        f" investigate the retrieval change and fix it."
    )


def test_citation_precision_meets_threshold() -> None:
    result = _latest_result("full")
    assert result["citation_precision"] >= CITATION_PRECISION_THRESHOLD, (
        f"Citation precision regressed to {result['citation_precision']:.3f}, below "
        f"{CITATION_PRECISION_THRESHOLD}. This should be structurally impossible (citations "
        f".validate_citations always strips invalid ids from the final Answer) -- if this "
        f"fails, something is bypassing that validation."
    )


def test_refusal_accuracy_meets_threshold() -> None:
    result = _latest_result("full")
    assert result["refusal_accuracy"] >= REFUSAL_ACCURACY_THRESHOLD, (
        f"Refusal accuracy regressed to {result['refusal_accuracy']:.3f}, below the "
        f"{REFUSAL_ACCURACY_THRESHOLD} threshold."
    )


def test_faithfulness_meets_threshold() -> None:
    result = _latest_result("full")
    if "faithfulness" not in result:
        pytest.skip("Latest full eval run used --no-judge; no faithfulness score to check.")
    assert result["faithfulness"] >= FAITHFULNESS_THRESHOLD, (
        f"Faithfulness regressed to {result['faithfulness']:.3f}, below the "
        f"{FAITHFULNESS_THRESHOLD} threshold."
    )
