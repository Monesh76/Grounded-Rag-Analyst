"""Parse 10-K HTML into sections keyed by Item number.

10-K HTML varies a lot by filer. Two heading styles are handled:

1. **By Item number** (the normal case): a short block starting with
   "Item 7." or "Item 7A." etc.
2. **By section name** (seen in some bank filings, which package their 10-K
   inside a full annual report and use plain headings like "Management's
   Discussion and Analysis" with no "Item 7" prefix): matched against a small
   table of known section names.

A table-of-contents table and inline cross-references ("see Item 7 below")
are not treated as headings -- see `_match_heading` for why.
"""

import re
import warnings
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from filings_rag.ingest.models import Section

# Real 10-Ks often carry an XML/iXBRL prolog even though the document is genuine
# (X)HTML we want parsed as HTML; bs4's heuristic mistakes that for a pure-XML file.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Elements whose own text we might treat as one block. A tag only becomes a leaf
# block if it has no *nested* tag from this set -- otherwise its child owns the text,
# which avoids collecting the same text twice (once for the div, once for the <p> inside it).
BLOCK_TAGS = ("p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6")

# Anchored to the start of the block: "Item 7." matches, but "See Item 7 for..." doesn't,
# because that block starts with "See". Requiring a separator right after the number/letter
# (not just a space) also rules out sentences like "Item 7 of this Report discusses...".
_ITEM_HEADING_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[.:—-]\s*(.*)$", re.IGNORECASE)

# canonical (item, title) pairs, also used as the name-based fallback for filings
# (mostly banks) that omit the "Item N" prefix on their real section headings.
_KNOWN_SECTIONS: list[tuple[str, str, str]] = [
    # (normalized name to match, item code, canonical title)
    ("business", "1", "Business"),
    ("risk factors", "1A", "Risk Factors"),
    ("unresolved staff comments", "1B", "Unresolved Staff Comments"),
    ("properties", "2", "Properties"),
    ("legal proceedings", "3", "Legal Proceedings"),
    (
        "managements discussion and analysis",
        "7",
        "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    ),
    (
        "quantitative and qualitative disclosures about market risk",
        "7A",
        "Quantitative and Qualitative Disclosures About Market Risk",
    ),
    (
        "financial statements and supplementary data",
        "8",
        "Financial Statements and Supplementary Data",
    ),
    ("financial statements", "8", "Financial Statements and Supplementary Data"),
]
_CANONICAL_TITLES = {item: title for _, item, title in _KNOWN_SECTIONS}
# Only treated as a heading if short -- a real name-only heading is just a few words,
# not a full sentence that happens to start with those words.
_MAX_FALLBACK_HEADING_LENGTH = 100
_STARTSWITH_SAFE_MIN_LENGTH = 20

_PAGE_BREAK_BEFORE_RE = re.compile(r"page-break-before\s*:\s*always", re.IGNORECASE)
_PAGE_BREAK_AFTER_RE = re.compile(r"page-break-after\s*:\s*always", re.IGNORECASE)


@dataclass
class _Block:
    text: str
    page: int
    is_table: bool = False


@dataclass
class _RawSection:
    item: str
    title: str
    page: int
    heading_text: str  # the raw block text that started this section
    body_parts: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n\n".join(self.body_parts)


def parse_filing_html(html: str | bytes) -> list[Section]:
    """Parse one 10-K's HTML into sections, deduplicated to one per Item.

    If an item's heading appears more than once (e.g. a short cross-reference
    stub plus the real content elsewhere), the occurrence with the most body
    text wins -- see the module docstring for why that happens with bank filings.
    """
    soup = BeautifulSoup(html, "lxml")
    blocks = _extract_blocks(soup)

    raw_by_item: dict[str, list[_RawSection]] = {}
    current: _RawSection | None = None
    for block in blocks:
        heading_text = block.text.strip()
        heading = None if block.is_table else _match_heading(heading_text)
        if heading:
            if current is not None and current.heading_text == heading_text:
                # A running header repeating verbatim on every page of a long
                # section (seen in Bank of America's and Goldman Sachs' filings)
                # -- without this check, each page would restart a fresh section,
                # fragmenting one long section into many one-page ones, of which
                # the keep-the-longest dedup below would only keep a single page.
                # Matched on the exact heading text, not just the item number: a
                # short cross-reference stub followed by the real content under a
                # *differently worded* heading for the same item (PLAN.md's other
                # bank-filing case) must still be treated as separate candidates.
                continue
            item, title = heading
            current = _RawSection(
                item=item,
                title=title or _CANONICAL_TITLES.get(item, ""),
                page=block.page,
                heading_text=heading_text,
            )
            raw_by_item.setdefault(item, []).append(current)
        elif current is not None:
            current.body_parts.append(block.text)

    winners = [max(candidates, key=lambda s: len(s.body)) for candidates in raw_by_item.values()]
    winners.sort(key=lambda s: s.page)
    return [Section(item=w.item, title=w.title, page=w.page, text=w.body) for w in winners]


def _extract_blocks(soup: BeautifulSoup) -> list[_Block]:
    page = 1
    blocks: list[_Block] = []
    for el in soup.find_all(True):
        style = el.get("style", "") or ""
        if _PAGE_BREAK_BEFORE_RE.search(style):
            page += 1

        if el.name == "table":
            rows = _table_rows(el)
            single_row_text = " ".join(rows[0]) if len(rows) == 1 else None
            # Many filers lay out real headings as a one-row, two-cell table --
            # "Item 7." in one cell, the title in the other -- rather than a <p>/<div>.
            # Only treat a table as a heading candidate when it has exactly one
            # content row (after dropping empty spacer cells); a multi-row table is
            # always data (this is what keeps the table of contents from matching,
            # since it has one row per item).
            if single_row_text and _match_heading(single_row_text):
                blocks.append(_Block(text=single_row_text, page=page))
            elif rows:
                blocks.append(_Block(text=_rows_to_markdown(rows), page=page, is_table=True))
        elif el.name in BLOCK_TAGS and not el.find(BLOCK_TAGS + ("table",)):
            text = el.get_text(" ", strip=True)
            if text:
                blocks.append(_Block(text=text, page=page))

        if _PAGE_BREAK_AFTER_RE.search(style):
            page += 1
    return blocks


def _table_rows(table: Tag) -> list[list[str]]:
    """Return each row's non-empty cell texts. Empty spacer cells (common in 10-K
    tables, used only to set column widths) are dropped, so a row that's purely
    layout comes back as an empty list and doesn't count as content."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    return rows


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Render table rows as pipe-separated lines.

    This is deliberately lossy (columns don't line up like a strict grid) --
    good enough to give an LLM the numbers in context, not meant for
    programmatic column math.
    """
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _match_heading(text: str) -> tuple[str, str] | None:
    text = text.strip()

    if m := _ITEM_HEADING_RE.match(text):
        return m.group(1).upper(), m.group(2).strip()

    if len(text) <= _MAX_FALLBACK_HEADING_LENGTH:
        normalized = _normalize(text)
        for name, item, title in _KNOWN_SECTIONS:
            if normalized == name:
                return item, title
            # A long, distinctive name (the full canonical title, roughly) is
            # also matched as a prefix -- a real filer's block is sometimes the
            # short name alone ("Risk Factors") and sometimes the full title
            # ("...Discussion and Analysis of Financial Condition and Results of
            # Operations"). A short, generic name (e.g. "business") stays
            # exact-match only: prefix-matching it risks misfiring on an
            # unrelated subsection that happens to start with the same common
            # word (seen for real: Bank of America has an MD&A subsection
            # literally titled "Business Segment Operations").
            if len(name) >= _STARTSWITH_SAFE_MIN_LENGTH and normalized.startswith(name):
                return item, title

    return None


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation/digits, collapse whitespace -- for name matching only."""
    text = re.sub(r"[^a-z\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()
