"""Tests for the 10-K HTML parser. All offline, using the saved fixture below."""

from pathlib import Path

import pytest

from filings_rag.ingest.models import Section
from filings_rag.ingest.parse import parse_filing_html

FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.htm"


@pytest.fixture
def sections() -> list[Section]:
    return parse_filing_html(FIXTURE.read_bytes())


def by_item(sections: list[Section], item: str) -> Section:
    matches = [s for s in sections if s.item == item]
    assert len(matches) == 1, f"expected exactly one section for Item {item}, got {len(matches)}"
    return matches[0]


def test_finds_all_expected_items(sections: list[Section]) -> None:
    assert {s.item for s in sections} == {"1", "1A", "7", "8"}


def test_ignores_table_of_contents(sections: list[Section]) -> None:
    # The TOC table lists every item with a page number; if it were read as headings
    # we'd see duplicate, low-content sections for each item.
    item_1 = by_item(sections, "1")
    assert "3" not in item_1.text  # the TOC's page-number column
    assert "design, manufacture and market" in item_1.text


def test_ignores_inline_cross_reference(sections: list[Section]) -> None:
    # "Additional discussion ... in Item 7 below" must stay part of Item 1's body,
    # not be mistaken for the start of Item 7.
    item_1 = by_item(sections, "1")
    assert "Item 7 below" in item_1.text
    item_7 = by_item(sections, "7")
    assert "Item 7 below" not in item_7.text


def test_captures_title_from_item_heading(sections: list[Section]) -> None:
    item_1a = by_item(sections, "1A")
    assert item_1a.title == "Risk Factors"


def test_table_content_kept_as_markdown(sections: list[Section]) -> None:
    item_7 = by_item(sections, "7")
    assert "| 2024 | $128.0B | $49.6B |" in item_7.text


def test_dedupes_duplicate_item_keeping_longest(sections: list[Section]) -> None:
    # Item 7 appears twice: a short "incorporated by reference" stub, and the real
    # MD&A content (heading-by-name, bank style). The stub must lose.
    item_7 = by_item(sections, "7")
    assert "Incorporated herein by reference" not in item_7.text
    assert "Net income was $49.6 billion" in item_7.text


def test_bank_style_heading_matched_by_name(sections: list[Section]) -> None:
    # "Management's Discussion and Analysis" has no "Item 7" prefix; matched via
    # the section-name fallback and given the canonical title.
    item_7 = by_item(sections, "7")
    assert item_7.title == (
        "Management's Discussion and Analysis of Financial Condition and Results of Operations"
    )


def test_pages_increase_in_document_order(sections: list[Section]) -> None:
    ordered = sorted(sections, key=lambda s: s.item)  # "1" < "1A" < "7" < "8" alphabetically too
    pages = [s.page for s in ordered]
    assert pages == sorted(pages)
    assert len(set(pages)) > 1  # page breaks actually moved the counter


def test_name_fallback_requires_short_block() -> None:
    # A long sentence that happens to start with a known section name must not
    # be mistaken for a heading.
    html = (
        "<html><body>"
        "<p>Item 1. Business</p>"
        "<p>Business risk factors and properties are discussed in later sections "
        "of this report in more detail than we can summarize here today.</p>"
        "</body></html>"
    )
    sections = parse_filing_html(html)
    assert len(sections) == 1
    assert "discussed in later sections" in sections[0].text


def test_no_headings_returns_empty_list() -> None:
    assert parse_filing_html("<html><body><p>Nothing relevant here.</p></body></html>") == []


def test_heading_inside_single_row_layout_table_is_detected() -> None:
    # Real-world case (seen in Amazon's 10-K): the heading itself is a table used
    # purely for two-column layout, not a data table -- one content row, "Item 7."
    # in one cell and the title in the other, plus an empty spacer row for widths.
    html = """
    <html><body>
    <table><tr><td></td><td></td></tr></table>
    <table>
      <tr><td style="width:1%"></td><td style="width:9%"></td></tr>
      <tr><td>Item&#160;7.</td><td>Management's Discussion and Analysis</td></tr>
    </table>
    <p>Revenue grew 11% year over year.</p>
    </body></html>
    """
    sections = parse_filing_html(html)
    assert len(sections) == 1
    assert sections[0].item == "7"
    assert "Revenue grew 11%" in sections[0].text


def test_multi_row_table_with_item_like_rows_is_not_treated_as_headings() -> None:
    # A table-of-contents built as a real <table> (not the fixture's cover page) with
    # one row per item must stay data, not spawn a "heading" per row.
    html = """
    <html><body>
    <table>
      <tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
      <tr><td>Item 1A.</td><td>Risk Factors</td><td>5</td></tr>
    </table>
    <p>Item 1. Business</p>
    <p>We build things.</p>
    </body></html>
    """
    sections = parse_filing_html(html)
    assert [s.item for s in sections] == ["1"]
    assert "3" not in sections[0].text


def test_running_header_repeat_is_a_continuation_not_a_new_section() -> None:
    # Real-world case (seen in Bank of America's and Goldman Sachs' 10-Ks): the
    # section heading repeats as a running header on every page of a long
    # section. Without treating a same-item repeat as a continuation, each page
    # restarts a fresh section and the keep-the-longest dedup would discard all
    # but a single page's worth of the real content.
    html = """
    <html><body>
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>Page one content about revenue growth.</p>
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>Page two content about operating expenses.</p>
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>Page three content about liquidity.</p>
    <p>Item 8. Financial Statements</p>
    <p>See accompanying statements.</p>
    </body></html>
    """
    sections = parse_filing_html(html)
    item_7 = by_item(sections, "7")
    assert "revenue growth" in item_7.text
    assert "operating expenses" in item_7.text
    assert "liquidity" in item_7.text
    # The repeated header itself shouldn't leak into the body text.
    assert item_7.text.count("Management's Discussion and Analysis") == 0


def test_long_canonical_title_matches_via_fallback_without_item_prefix() -> None:
    # Real-world case: a heading using the full canonical title with no "Item 7"
    # prefix at all (distinct from the short-name fallback already covered
    # elsewhere in this file).
    html = (
        "<html><body>"
        "<p>Management's Discussion and Analysis of Financial Condition "
        "and Results of Operations</p>"
        "<p>Revenue increased due to strong demand.</p>"
        "</body></html>"
    )
    sections = parse_filing_html(html)
    assert len(sections) == 1
    assert sections[0].item == "7"
    assert "Revenue increased" in sections[0].text


def test_short_name_fallback_does_not_match_an_unrelated_subsection() -> None:
    # Real-world case: Bank of America's MD&A has a subsection literally titled
    # "Business Segment Operations" -- must not be mistaken for Item 1 (whose
    # fallback name is just "Business") since it's a different topic entirely.
    html = (
        "<html><body>"
        "<p>Item 1. Business</p>"
        "<p>We design, manufacture and sell products.</p>"
        "<p>Business Segment Operations</p>"
        "<p>This subsection is about internal segment reporting, not Item 1.</p>"
        "</body></html>"
    )
    sections = parse_filing_html(html)
    assert [s.item for s in sections] == ["1"]
    assert "internal segment reporting" in sections[0].text  # fell through as body
