"""Application settings, loaded from environment variables and `.env`.

All configuration goes through `Settings` so nothing (URLs, model names, keys,
thresholds) is hard-coded in logic. Environment variables take precedence over
values in `.env`.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env also holds POSTGRES_* vars meant for docker-compose
    )

    # Default matches docker-compose.yml so a fresh clone works without a .env.
    database_url: str = "postgresql://filings:filings@localhost:5432/filings"


@lru_cache
def get_settings() -> Settings:
    """Return one shared Settings instance (call `get_settings.cache_clear()` in tests)."""
    return Settings()
