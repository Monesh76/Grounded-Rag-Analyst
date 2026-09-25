from pathlib import Path

import pytest

from filings_rag.config import Settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each test in an empty dir so the developer's real .env and env vars don't leak in."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)


def test_default_database_url() -> None:
    assert Settings().database_url == "postgresql://filings:filings@localhost:5432/filings"


def test_reads_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql://u:p@dotenv:5432/db\n")
    assert Settings().database_url == "postgresql://u:p@dotenv:5432/db"


def test_env_var_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql://u:p@dotenv:5432/db\n")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@envvar:5432/db")
    assert Settings().database_url == "postgresql://u:p@envvar:5432/db"
