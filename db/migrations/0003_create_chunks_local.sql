-- chunks_local: same shape as chunks, but embedding is vector(384) --
-- sized for the local sentence-transformers embedder (BAAI/bge-small-en-v1.5),
-- not Voyage. CI's make eval-ci uses this table: dense retrieval always has to
-- embed the incoming question at query time, every call, regardless of
-- whether the corpus is pre-embedded -- a real gap found when CI failed with
-- "VOYAGE_API_KEY must be set" even though the corpus fixture never needed
-- re-embedding. Using the local embedder for both corpus and query keeps
-- eval-ci at zero API calls, genuinely, not just for the corpus.

CREATE TABLE IF NOT EXISTS chunks_local (
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
    embedding vector(384) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_local_embedding_hnsw_idx
    ON chunks_local USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_local_tsv_gin_idx
    ON chunks_local USING gin (tsv);

CREATE INDEX IF NOT EXISTS chunks_local_ticker_fiscal_year_idx
    ON chunks_local (ticker, fiscal_year);
