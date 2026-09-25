-- chunks: one row per chunk, with everything retrieval needs (dense + keyword + filters)
-- in a single table, so hybrid search is plain SQL rather than a join across systems.
--
-- embedding is sized for voyage-finance-2 (1024 dims), the production embedder chosen
-- in P2. pgvector requires a fixed dimension per column, so comparing against the local
-- sentence-transformers embedder (384 dims) would need its own column or table -- not
-- done here since PLAN.md's schema calls for one `embedding` column; see
-- docs/decisions/003-chunk-embed-load.md.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id text PRIMARY KEY,                    -- deterministic: TICKER_FISCALYEAR_ITEM_INDEX
    doc_id text NOT NULL,                   -- filing's accession number
    ticker text NOT NULL,
    company text NOT NULL,
    fiscal_year integer NOT NULL,
    item text NOT NULL,                     -- "7", "1A", ...
    section_title text NOT NULL,
    page integer NOT NULL,                  -- approximate: the section's start page
    chunk_index integer NOT NULL,           -- position within the section, from 0
    text text NOT NULL,
    embedding vector(1024) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

-- Cosine distance matches how most embedding APIs (including Voyage) recommend
-- comparing their vectors.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin_idx
    ON chunks USING gin (tsv);

-- Supports P3's company/fiscal-year filters without a sequential scan.
CREATE INDEX IF NOT EXISTS chunks_ticker_fiscal_year_idx
    ON chunks (ticker, fiscal_year);
