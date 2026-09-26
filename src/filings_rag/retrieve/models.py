from pydantic import BaseModel


class Filters(BaseModel):
    """Metadata filters extracted from a question, applied as a WHERE clause."""

    ticker: str | None = None
    fiscal_year: int | None = None


class RetrievalResult(BaseModel):
    id: str
    doc_id: str
    ticker: str
    company: str
    fiscal_year: int
    item: str
    section_title: str
    page: int
    text: str
    score: float  # meaning depends on the stage: distance, ts_rank, RRF score, or
    # reranker score -- only comparable to other scores from the same stage.
