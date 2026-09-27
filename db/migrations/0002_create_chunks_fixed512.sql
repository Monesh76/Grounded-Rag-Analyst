-- chunks_fixed512: same schema as chunks, populated by P6 experiment A's
-- fixed-512-token chunking (ingest/chunk.py's chunk_filing_fixed) instead of
-- the production section-aware chunker. A separate table, not a column on
-- chunks, so experiment A's data can't collide with or overwrite production
-- rows sharing the same deterministic chunk id -- see docs/decisions/007.

CREATE TABLE IF NOT EXISTS chunks_fixed512 (
    id text PRIMARY KEY,
    doc_id text NOT NULL,
    ticker text NOT NULL,
    company text NOT NULL,
    fiscal_year integer NOT NULL,
    item text NOT NULL,
    section_title text NOT NULL,
    page integer NOT NULL,
    chunk_index integer NOT NULL,
    text text NOT NULL,
    embedding vector(1024) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_fixed512_embedding_hnsw_idx
    ON chunks_fixed512 USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_fixed512_tsv_gin_idx
    ON chunks_fixed512 USING gin (tsv);

CREATE INDEX IF NOT EXISTS chunks_fixed512_ticker_fiscal_year_idx
    ON chunks_fixed512 (ticker, fiscal_year);
