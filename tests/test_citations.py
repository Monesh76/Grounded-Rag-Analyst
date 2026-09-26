from filings_rag.generate.citations import validate_citations
from filings_rag.retrieve.models import RetrievalResult


def chunk(id: str) -> RetrievalResult:
    return RetrievalResult(
        id=id,
        doc_id="doc",
        ticker="AAPL",
        company="Apple",
        fiscal_year=2024,
        item="7",
        section_title="MD&A",
        page=23,
        text=f"text for {id}",
        score=0.9,
    )


def test_valid_citation_is_kept_and_becomes_a_source() -> None:
    result = validate_citations("Revenue grew 5% [c:AAPL_2024_7_0].", [chunk("AAPL_2024_7_0")])
    assert result.text == "Revenue grew 5% [c:AAPL_2024_7_0]."
    assert [s.id for s in result.sources] == ["AAPL_2024_7_0"]
    assert result.has_invalid_citations is False


def test_invalid_citation_is_stripped_and_flagged() -> None:
    result = validate_citations("Revenue grew 5% [c:made_up_id].", [chunk("AAPL_2024_7_0")])
    assert "[c:made_up_id]" not in result.text
    assert result.sources == []
    assert result.invalid_ids == ["made_up_id"]
    assert result.has_invalid_citations is True


def test_mixed_valid_and_invalid_citations() -> None:
    text = "First claim [c:real_1]. Second, unsupported claim [c:fake_1]."
    result = validate_citations(text, [chunk("real_1")])
    assert "[c:real_1]" in result.text
    assert "[c:fake_1]" not in result.text
    assert [s.id for s in result.sources] == ["real_1"]
    assert result.invalid_ids == ["fake_1"]


def test_repeated_citation_to_same_id_produces_one_source() -> None:
    text = "Claim one [c:x]. Claim two, same source [c:x]."
    result = validate_citations(text, [chunk("x")])
    assert len(result.sources) == 1


def test_sources_are_in_first_citation_order() -> None:
    text = "[c:second] then [c:first]"
    result = validate_citations(text, [chunk("first"), chunk("second")])
    assert [s.id for s in result.sources] == ["second", "first"]


def test_no_citations_returns_text_unchanged_and_no_sources() -> None:
    result = validate_citations("Not found in the filings.", [chunk("a")])
    assert result.text == "Not found in the filings."
    assert result.sources == []
    assert result.has_invalid_citations is False


def test_source_carries_full_chunk_metadata() -> None:
    result = validate_citations("[c:AAPL_2024_7_0]", [chunk("AAPL_2024_7_0")])
    source = result.sources[0]
    assert source.ticker == "AAPL"
    assert source.company == "Apple"
    assert source.fiscal_year == 2024
    assert source.item == "7"
    assert source.section_title == "MD&A"
    assert source.page == 23
