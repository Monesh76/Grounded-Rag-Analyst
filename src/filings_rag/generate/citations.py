"""Validates every `[c:<id>]` citation in an LLM answer against the chunks that
were actually retrieved, so a fabricated citation (an id the model made up, or
copied from its own training data instead of the provided context) can never
reach the user looking like a real source.
"""

import re

from pydantic import BaseModel

from filings_rag.generate.models import Source
from filings_rag.retrieve.models import RetrievalResult

_CITATION_RE = re.compile(r"\[c:([^\]]+)\]")


class CitationResult(BaseModel):
    text: str  # answer text with invalid citations stripped out
    sources: list[Source]  # one per valid cited id, in first-citation order
    invalid_ids: list[str]  # ids the model cited that weren't in the retrieved set

    @property
    def has_invalid_citations(self) -> bool:
        return bool(self.invalid_ids)


def find_citation_ids(text: str) -> list[str]:
    """All `[c:<id>]` ids mentioned in `text`, in order, duplicates included.
    Used by the eval harness to measure citation precision against the model's
    *raw* output before validate_citations() strips anything invalid.
    """
    return [m.group(1) for m in _CITATION_RE.finditer(text)]


def validate_citations(text: str, retrieved: list[RetrievalResult]) -> CitationResult:
    by_id = {r.id: r for r in retrieved}
    sources: list[Source] = []
    seen_valid: set[str] = set()
    invalid_ids: list[str] = []
    seen_invalid: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        cid = match.group(1)
        chunk = by_id.get(cid)
        if chunk is None:
            if cid not in seen_invalid:
                seen_invalid.add(cid)
                invalid_ids.append(cid)
            return ""  # strip: never show a citation to a chunk that wasn't retrieved

        if cid not in seen_valid:
            seen_valid.add(cid)
            sources.append(
                Source(
                    id=chunk.id,
                    ticker=chunk.ticker,
                    company=chunk.company,
                    fiscal_year=chunk.fiscal_year,
                    item=chunk.item,
                    section_title=chunk.section_title,
                    page=chunk.page,
                )
            )
        return match.group(0)  # keep valid citations as-is

    cleaned = _CITATION_RE.sub(replace, text)
    # A stripped citation can leave "claim  ." or "claim ." behind; tidy the
    # obvious cases without trying to fully re-flow the sentence.
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    cleaned = re.sub(r" ([.,;:])", r"\1", cleaned)

    return CitationResult(text=cleaned, sources=sources, invalid_ids=invalid_ids)
