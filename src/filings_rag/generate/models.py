from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class Source(BaseModel):
    """One chunk the answer actually cited, with enough metadata to show a user
    where the claim came from."""

    id: str
    ticker: str
    company: str
    fiscal_year: int
    item: str
    section_title: str
    page: int


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool
    latency_ms: float
    tokens: int
    cost_usd: float
