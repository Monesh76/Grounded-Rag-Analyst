# Builds the FastAPI service only. The Streamlit UI (ui/app.py) is deployed
# separately on Streamlit Community Cloud, which runs it directly from the
# repo without a Dockerfile -- see docs/deployment.md.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY src/ src/
COPY db/ db/
RUN uv sync --locked --no-dev

FROM python:3.12-slim AS runtime

# --create-home matters: sentence-transformers/huggingface_hub cache the
# reranker model under $HOME/.cache on first use, and a homeless user (found
# by actually running this image, not by reading it) fails with a
# PermissionError trying to mkdir under a nonexistent /home/app.
RUN groupadd --system app && useradd --system --create-home --gid app app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/db /app/db

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

CMD ["uvicorn", "filings_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
