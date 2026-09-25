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
