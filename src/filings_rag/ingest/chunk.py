"""Section-aware chunking.

Each section (as parsed by `parse.py`) is a sequence of blocks -- paragraphs and
tables -- separated by blank lines. Chunking packs those blocks greedily up to
`chunk_max_tokens`, never crosses a section boundary (each section is chunked
independently), and repeats the last ~`chunk_overlap_ratio` of a chunk at the
start of the next one so nearby chunks share context.

Token counts are a rough estimate (~4/3 tokens per word), not a real tokenizer --
good enough for chunk-size boundaries, and keeps this offline and dependency-free
(a real tokenizer, e.g. tiktoken, needs a network call to download its encoding
file on first use, which would break running tests with no network).
"""

import re
from dataclasses import dataclass

from filings_rag.config import Settings
from filings_rag.ingest.models import Chunk, FilingRef, Section

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 4 / 3))


def _is_table_block(block: str) -> bool:
    return block.lstrip().startswith("|")


@dataclass
class _Draft:
    """A chunk being assembled, before it's turned into a Chunk with an id."""

    blocks: list[str]

    @property
    def tokens(self) -> int:
        return sum(estimate_tokens(b) for b in self.blocks)

    @property
    def text(self) -> str:
        return "\n\n".join(self.blocks)


def chunk_section(section: Section, settings: Settings) -> list[str]:
    """Split one section's text into chunk texts. Blocks are the paragraphs and
    tables `parse.py` joined with blank lines; splitting on them recovers the
    original units without needing to change how sections are stored."""
    blocks = [b for b in section.text.split("\n\n") if b.strip()]
    chunks: list[_Draft] = []
    current = _Draft(blocks=[])

    for block in blocks:
        block_tokens = estimate_tokens(block)

        if block_tokens > settings.chunk_max_tokens:
            if current.blocks:
                chunks.append(current)
                current = _Draft(blocks=[])
            chunks.extend(_Draft(blocks=[b]) for b in _split_oversized_block(block, settings))
            continue

        if current.blocks and current.tokens + block_tokens > settings.chunk_max_tokens:
            chunks.append(current)
            current = _Draft(blocks=_overlap_tail(current.blocks, settings))

        current.blocks.append(block)

    if current.blocks:
        chunks.append(current)

    return [c.text for c in chunks]


def _overlap_tail(blocks: list[str], settings: Settings) -> list[str]:
    """The trailing blocks of a just-closed chunk, to seed the next one with
    shared context. Tables are skipped -- they're often large on their own, and
    duplicating one into the next chunk just to hit the overlap ratio would push
    that chunk well past the size target."""
    target = round(settings.chunk_max_tokens * settings.chunk_overlap_ratio)
    tail: list[str] = []
    tokens = 0
    for block in reversed(blocks):
        if _is_table_block(block) or tokens >= target:
            break
        tail.insert(0, block)
        tokens += estimate_tokens(block)
    return tail


def _split_oversized_block(block: str, settings: Settings) -> list[str]:
    """A single block bigger than chunk_max_tokens on its own.

    Tables are kept whole even when oversized -- splitting mid-table would produce
    a chunk of orphaned rows with no header, which is worse for an LLM than one
    chunk that runs long. A paragraph is split by sentence instead, since that's
    safe to do without losing meaning.
    """
    if _is_table_block(block):
        return [block]

    sentences = _SENTENCE_SPLIT_RE.split(block)
    parts: list[str] = []
    current: list[str] = []
    tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current and tokens + sentence_tokens > settings.chunk_max_tokens:
            parts.append(" ".join(current))
            current = []
            tokens = 0
        current.append(sentence)
        tokens += sentence_tokens
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_filing(filing: FilingRef, sections: list[Section], settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        for index, text in enumerate(chunk_section(section, settings)):
            chunks.append(
                Chunk(
                    id=f"{filing.ticker}_{filing.fiscal_year}_{section.item}_{index}",
                    doc_id=filing.accession_number,
                    ticker=filing.ticker,
                    company=filing.company,
                    fiscal_year=filing.fiscal_year,
                    item=section.item,
                    section_title=section.title,
                    page=section.page or 1,
                    chunk_index=index,
                    text=text,
                )
            )
    return chunks
