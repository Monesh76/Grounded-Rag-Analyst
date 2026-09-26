"""Application settings, loaded from environment variables and `.env`.

All configuration goes through `Settings` so nothing (URLs, model names, keys,
thresholds) is hard-coded in logic. Environment variables take precedence over
values in `.env`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env also holds POSTGRES_* vars meant for docker-compose
    )

    # Default matches docker-compose.yml so a fresh clone works without a .env.
    database_url: str = "postgresql://filings:filings@localhost:5432/filings"

    # A separate database for integration tests, on the same Postgres instance/
    # container. Tests must never point at `database_url`: several (test_db.py's
    # migration tests, in particular) DROP TABLE as part of setup, which would
    # silently destroy real loaded data if it ran against the same database
    # `make load` populates -- a mistake this project made once already.
    test_database_url: str = "postgresql://filings:filings@localhost:5432/filings_test"

    # --- SEC EDGAR ingestion ---
    # SEC's fair-access policy requires "Name email" in the User-Agent. Optional here so
    # the rest of the app works without it; the EDGAR client refuses to start if it's missing.
    sec_user_agent: str | None = None
    # SEC allows ~10 req/s; we stay below that on purpose.
    sec_max_requests_per_second: float = Field(default=8.0, gt=0, le=10)
    sec_max_attempts: int = Field(default=4, ge=1)
    sec_backoff_seconds: float = Field(default=1.0, ge=0)
    sec_timeout_seconds: float = Field(default=30.0, gt=0)
    filings_per_company: int = Field(default=3, ge=1)

    # --- Chunking (P2) ---
    chunk_min_tokens: int = Field(default=500, gt=0)
    chunk_max_tokens: int = Field(default=800, gt=0)
    chunk_overlap_ratio: float = Field(default=0.15, ge=0, lt=1)

    # --- Embedding (P2) ---
    # Which Embedder `get_embedder()` builds. "voyage" is what the chunks table's
    # embedding column is sized for (vector(1024)); switching to "local" (384 dims)
    # would need its own column or table -- see docs/decisions/003.
    embedder_provider: Literal["voyage", "local"] = "voyage"
    # Hosted: Voyage AI. Chosen as Anthropic's recommended embedding partner, with a
    # finance-tuned model that fits this project's domain.
    voyage_api_key: str | None = None
    voyage_model: str = "voyage-finance-2"
    # PLACEHOLDER -- verify against https://docs.voyageai.com/docs/pricing before
    # trusting the cost estimate `make load` prints; pricing changes over time.
    voyage_price_per_million_tokens: float = 0.12
    voyage_batch_size: int = Field(default=100, gt=0)
    voyage_max_requests_per_second: float = Field(default=4.0, gt=0)
    voyage_max_attempts: int = Field(default=4, ge=1)
    voyage_backoff_seconds: float = Field(default=1.0, ge=0)
    voyage_timeout_seconds: float = Field(default=60.0, gt=0)

    # Local: sentence-transformers, runs on CPU, no API cost.
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_embedding_batch_size: int = Field(default=32, gt=0)

    # --- Retrieval (P3) ---
    retrieval_dense_top_k: int = Field(default=50, gt=0)
    retrieval_keyword_top_k: int = Field(default=50, gt=0)
    retrieval_rrf_k: int = Field(default=60, gt=0)  # reciprocal rank fusion constant
    retrieval_rerank_top_k: int = Field(default=6, gt=0)
    reranker_model: str = "BAAI/bge-reranker-base"

    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"


@lru_cache
def get_settings() -> Settings:
    """Return one shared Settings instance (call `get_settings.cache_clear()` in tests)."""
    return Settings()
