"""Load chunked, embedded filings into Postgres (`make load`).

Idempotent: rows are upserted by their deterministic chunk id (see chunk.py), so
re-running with an unchanged manifest/parsed output and the same embedder replaces
identical rows rather than duplicating them.
"""

import argparse
import sys
from collections.abc import Callable

from pgvector.psycopg import register_vector

from filings_rag.config import Settings, get_settings
from filings_rag.db import get_connection, run_migrations
from filings_rag.embed import Embedder, get_embedder
from filings_rag.ingest.chunk import chunk_filing, chunk_filing_fixed
from filings_rag.ingest.models import Chunk, FilingRef, Manifest, ParsedFiling, Section
from filings_rag.retrieve.common import CHUNK_TABLES

ChunkFn = Callable[[FilingRef, list[Section], Settings], list[Chunk]]


def _upsert_sql(table: str) -> str:
    if table not in CHUNK_TABLES:
        raise ValueError(f"Unknown chunk table {table!r}; choose one of {CHUNK_TABLES}")
    return f"""
    INSERT INTO {table} (id, doc_id, ticker, company, fiscal_year, item, section_title,
                         page, chunk_index, text, embedding)
    VALUES (%(id)s, %(doc_id)s, %(ticker)s, %(company)s, %(fiscal_year)s, %(item)s,
            %(section_title)s, %(page)s, %(chunk_index)s, %(text)s, %(embedding)s)
    ON CONFLICT (id) DO UPDATE SET
        doc_id = EXCLUDED.doc_id,
        ticker = EXCLUDED.ticker,
        company = EXCLUDED.company,
        fiscal_year = EXCLUDED.fiscal_year,
        item = EXCLUDED.item,
        section_title = EXCLUDED.section_title,
        page = EXCLUDED.page,
        chunk_index = EXCLUDED.chunk_index,
        text = EXCLUDED.text,
        embedding = EXCLUDED.embedding
    """  # noqa: S608 -- table comes from our own fixed, validated allow-list


def load_all(
    settings: Settings,
    embedder: Embedder | None = None,
    table: str = "chunks",
    chunk_fn: ChunkFn = chunk_filing,
) -> dict:
    embedder = embedder or get_embedder(settings)
    conn = get_connection(settings)
    register_vector(conn)  # lets psycopg send/receive python lists as pgvector's type
    run_migrations(conn)

    manifest = Manifest.model_validate_json(settings.manifest_path.read_text())
    filings_loaded = 0
    chunks_loaded = 0
    upsert_sql = _upsert_sql(table)

    for entry in manifest.filings:
        parsed_path = settings.parsed_dir / f"{entry.ticker}_{entry.fiscal_year}.json"
        if not parsed_path.exists():
            continue  # parsed output is gitignored/regenerable; skip if missing rather than fail

        parsed = ParsedFiling.model_validate_json(parsed_path.read_text())
        chunks = chunk_fn(parsed.filing, parsed.sections, settings)
        if not chunks:
            continue

        vectors = embedder.embed([c.text for c in chunks])
        rows = [
            {**c.model_dump(), "embedding": vector}
            for c, vector in zip(chunks, vectors, strict=True)
        ]
        with conn.cursor() as cur:
            cur.executemany(upsert_sql, rows)
        conn.commit()

        filings_loaded += 1
        chunks_loaded += len(chunks)

    (total_in_db,) = conn.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
    conn.close()

    tokens_used = getattr(embedder, "total_tokens_used", 0)
    return {
        "filings_loaded": filings_loaded,
        "chunks_loaded": chunks_loaded,
        "total_chunks_in_db": total_in_db,
        "tokens_used": tokens_used,
        "estimated_cost_usd": tokens_used / 1_000_000 * settings.voyage_price_per_million_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m filings_rag.ingest.load")
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Load P6 experiment A's fixed-512-token chunking into chunks_fixed512 "
        "instead of the production section-aware chunker into chunks",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.fixed:
        summary = load_all(settings, table="chunks_fixed512", chunk_fn=chunk_filing_fixed)
    else:
        summary = load_all(settings)

    print(f"Filings loaded:          {summary['filings_loaded']}")
    print(f"Chunks loaded (this run): {summary['chunks_loaded']}")
    print(f"Chunks in database:      {summary['total_chunks_in_db']}")
    if summary["tokens_used"]:
        print(f"Embedding tokens used:   {summary['tokens_used']:,}")
        print(
            f"Estimated embedding cost: ${summary['estimated_cost_usd']:.4f} "
            "(PLACEHOLDER price -- verify against https://docs.voyageai.com/docs/pricing)"
        )
    else:
        print("Embedding cost:          $0 (local model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
