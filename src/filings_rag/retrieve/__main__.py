"""CLI: python -m filings_rag.retrieve "question" --mode hybrid_rerank

Prints the top results with scores and section labels, for spot-checking
retrieval quality by eye.
"""

import argparse
import sys

from filings_rag import tracing
from filings_rag.config import get_settings
from filings_rag.db import get_connection
from filings_rag.embed import get_embedder
from filings_rag.retrieve.models import RetrievalResult
from filings_rag.retrieve.rerank import Reranker
from filings_rag.retrieve.search import MODES, search


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m filings_rag.retrieve")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    conn = get_connection(settings)
    embedder = get_embedder(settings)
    reranker = Reranker.from_settings(settings) if args.mode == "hybrid_rerank" else None

    results = search(conn, embedder, reranker, args.question, args.mode, settings, args.top_k)
    _print_results(results)
    tracing.flush(settings)
    return 0


def _print_results(results: list[RetrievalResult]) -> None:
    if not results:
        print("No results.")
        return
    for i, r in enumerate(results, 1):
        print(
            f"{i}. [{r.ticker} FY{r.fiscal_year} Item {r.item}] {r.section_title} "
            f"(p.{r.page}, score={r.score:.4f})  {r.id}"
        )
        preview = r.text[:200].replace("\n", " ")
        print(f"   {preview}...")
        print()


if __name__ == "__main__":
    sys.exit(main())
