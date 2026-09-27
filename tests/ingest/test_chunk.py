from filings_rag.config import Settings
from filings_rag.ingest.chunk import (
    chunk_filing,
    chunk_filing_fixed,
    chunk_section,
    chunk_section_fixed,
    estimate_tokens,
)
from filings_rag.ingest.models import FilingRef, Section

FILING = FilingRef(
    ticker="AAPL",
    cik=320193,
    company="Apple",
    form="10-K",
    accession_number="0000320193-25-000079",
    filing_date="2025-10-31",
    report_date="2025-09-27",
    fiscal_year=2025,
    primary_document="aapl.htm",
    url="https://example.com/aapl.htm",
)


def words(n: int, prefix: str = "word") -> str:
    """A block of n space-separated words (deterministic size for tests)."""
    return " ".join(f"{prefix}{i}" for i in range(n))


def small_settings(**overrides) -> Settings:
    defaults = dict(chunk_min_tokens=20, chunk_max_tokens=40, chunk_overlap_ratio=0.2)
    return Settings(**{**defaults, **overrides})


def test_estimate_tokens_scales_with_word_count() -> None:
    assert estimate_tokens(words(30)) == round(30 * 4 / 3)


def test_packs_small_blocks_without_exceeding_max() -> None:
    settings = small_settings()
    # 6 blocks of ~10 tokens (7-8 words) each; max is 40 tokens.
    blocks = [words(8) for _ in range(6)]
    section = Section(item="1", title="Business", page=3, text="\n\n".join(blocks))

    chunks = chunk_section(section, settings)

    assert len(chunks) > 1  # didn't fit in one chunk
    for c in chunks:
        assert estimate_tokens(c) <= settings.chunk_max_tokens


def test_all_content_is_preserved_across_chunks() -> None:
    settings = small_settings()
    blocks = [f"paragraph-{i} " + words(8) for i in range(6)]
    section = Section(item="1", title="Business", page=3, text="\n\n".join(blocks))

    chunks = chunk_section(section, settings)
    combined = " ".join(chunks)
    for i in range(6):
        assert f"paragraph-{i}" in combined


def test_consecutive_chunks_overlap() -> None:
    settings = small_settings()
    blocks = [f"paragraph-{i} " + words(10) for i in range(5)]
    section = Section(item="1", title="Business", page=3, text="\n\n".join(blocks))

    chunks = chunk_section(section, settings)

    assert len(chunks) >= 2
    # The tail of chunk N should reappear at the start of chunk N+1.
    first_tail_block = chunks[0].split("\n\n")[-1]
    assert first_tail_block in chunks[1]


def test_table_kept_whole_even_when_oversized() -> None:
    settings = small_settings()
    table = "| " + " | ".join(f"cell{i}" for i in range(60)) + " |"  # way over 40 tokens
    section = Section(item="8", title="Financial Statements", page=10, text=table)

    chunks = chunk_section(section, settings)

    assert chunks == [table]  # never split, even though it's oversized


def test_table_does_not_grow_neighboring_chunks_via_overlap() -> None:
    settings = small_settings()
    table = "| " + " | ".join(f"cell{i}" for i in range(60)) + " |"
    next_para = words(10)
    section = Section(
        item="8", title="Financial Statements", page=10, text=f"{table}\n\n{next_para}"
    )

    chunks = chunk_section(section, settings)

    assert chunks[0] == table
    assert "|" not in chunks[1]  # table wasn't carried into the next chunk as overlap


def test_oversized_paragraph_is_split_by_sentence_not_dropped() -> None:
    settings = small_settings()
    sentences = [f"This is sentence number {i} with some extra words in it." for i in range(10)]
    paragraph = " ".join(sentences)
    section = Section(item="7", title="MD&A", page=20, text=paragraph)

    chunks = chunk_section(section, settings)

    assert len(chunks) > 1
    combined = " ".join(chunks)
    for i in range(10):
        assert f"sentence number {i}" in combined


def test_chunking_never_crosses_a_section_boundary() -> None:
    settings = small_settings()
    section_1 = Section(item="1", title="Business", page=3, text=words(30))
    section_7 = Section(item="7", title="MD&A", page=20, text=words(30))

    chunks = chunk_filing(FILING, [section_1, section_7], settings)

    assert {c.item for c in chunks} == {"1", "7"}
    for c in chunks:
        assert c.section_title == ("Business" if c.item == "1" else "MD&A")


def test_chunk_ids_are_deterministic_and_sequential() -> None:
    settings = small_settings()
    blocks = [words(8) for _ in range(6)]
    section = Section(item="1A", title="Risk Factors", page=5, text="\n\n".join(blocks))

    chunks = chunk_filing(FILING, [section], settings)

    assert [c.id for c in chunks] == [f"AAPL_2025_1A_{i}" for i in range(len(chunks))]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == FILING.accession_number for c in chunks)


# --- chunk_section_fixed / chunk_filing_fixed (P6 experiment A's baseline) ---


def test_fixed_chunking_splits_into_roughly_equal_word_windows() -> None:
    section = Section(item="7", title="MD&A", page=20, text=words(300))

    chunks = chunk_section_fixed(section, tokens=40)

    assert len(chunks) > 1
    words_per_chunk = round(40 * 3 / 4)
    for chunk in chunks[:-1]:  # the last window may be shorter
        assert len(chunk.split()) == words_per_chunk


def test_fixed_chunking_preserves_all_words_in_order() -> None:
    section = Section(item="7", title="MD&A", page=20, text=words(50))

    chunks = chunk_section_fixed(section, tokens=40)

    assert " ".join(chunks).split() == words(50).split()


def test_fixed_chunking_ignores_paragraph_boundaries() -> None:
    # Unlike chunk_section, chunk_section_fixed doesn't respect "\n\n" breaks --
    # that's the whole point of a blind baseline to compare section-aware
    # chunking against.
    section = Section(item="7", title="MD&A", page=20, text=f"{words(20)}\n\n{words(20)}")

    chunks = chunk_section_fixed(section, tokens=200)

    assert len(chunks) == 1
    assert "\n\n" not in chunks[0]


def test_chunk_filing_fixed_uses_fixed_chunk_tokens_setting() -> None:
    settings = small_settings(fixed_chunk_tokens=40)
    section = Section(item="1", title="Business", page=3, text=words(100))

    chunks = chunk_filing_fixed(FILING, [section], settings)

    assert len(chunks) > 1
    assert [c.id for c in chunks] == [f"AAPL_2025_1_{i}" for i in range(len(chunks))]
