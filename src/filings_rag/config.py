"""Application settings, loaded from environment variables and `.env`.

All configuration goes through `Settings` so nothing (URLs, model names, keys,
thresholds) is hard-coded in logic. Environment variables take precedence over
values in `.env`.
"""

from functools import lru_cache
from pathlib import Path

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
