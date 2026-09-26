"""FastAPI app: POST /ask (supports streaming), GET /health.

Citation validation and the evidence gate both need the LLM's *complete*
response (a fabricated citation can only be caught once the whole answer is
in hand), so a naive token-by-token pass-through from the LLM could show
unvalidated text before we could catch a problem. Instead: pipeline.ask()
already runs to completion and validates before this module ever sees the
answer; when `stream=true`, the endpoint streams that already-validated text
to the client in pieces, followed by one final JSON line with the structured
metadata -- "streams tokens" at the HTTP layer, without ever exposing content
that hasn't been checked.
"""

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from filings_rag import tracing
from filings_rag.config import Settings, get_settings
from filings_rag.generate.models import Answer
from filings_rag.pipeline import ask as pipeline_ask


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    tracing.flush(get_settings())


app = FastAPI(title="FilingsRAG", lifespan=_lifespan)

AskFn = Callable[[str, Settings], Answer]


def get_ask_fn() -> AskFn:
    """A seam for tests: override via app.dependency_overrides to inject a fake
    without touching the real DB, embedder, reranker or LLM."""
    return pipeline_ask


class AskRequest(BaseModel):
    question: str
    stream: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=None)  # return type is a union (Answer or a stream); see below
def ask(
    request: AskRequest,
    settings: Settings = Depends(get_settings),
    ask_fn: AskFn = Depends(get_ask_fn),
) -> Answer | StreamingResponse:
    answer = ask_fn(request.question, settings)

    if not request.stream:
        return answer

    return StreamingResponse(_stream(answer), media_type="text/event-stream")


def _stream(answer: Answer, chunk_size: int = 40) -> Iterator[str]:
    for i in range(0, len(answer.answer), chunk_size):
        yield answer.answer[i : i + chunk_size]
    yield "\n\n" + answer.model_dump_json()
