from filings_rag.generate.prompts import build_system_prompt, build_user_message
from filings_rag.retrieve.models import RetrievalResult


def chunk(id: str, text: str = "chunk text") -> RetrievalResult:
    return RetrievalResult(
        id=id,
        doc_id="doc",
        ticker="AAPL",
        company="Apple",
        fiscal_year=2024,
        item="7",
        section_title="MD&A",
        page=23,
        text=text,
        score=0.9,
    )


def test_system_prompt_includes_the_exact_refusal_text() -> None:
    prompt = build_system_prompt("Not found in the filings.")
    assert '"Not found in the filings."' in prompt


def test_system_prompt_instructs_citation_format() -> None:
    prompt = build_system_prompt("Not found in the filings.")
    assert "[c:<id>]" in prompt


def test_user_message_includes_question_and_chunk_ids() -> None:
    message = build_user_message("What was revenue?", [chunk("AAPL_2024_7_0")])
    assert "What was revenue?" in message
    assert "[c:AAPL_2024_7_0]" in message


def test_user_message_includes_chunk_metadata_and_text() -> None:
    message = build_user_message("q", [chunk("id1", text="Net revenue was $391B.")])
    assert "Apple" in message
    assert "FY2024" in message
    assert "Item 7" in message
    assert "p.23" in message
    assert "Net revenue was $391B." in message


def test_user_message_includes_every_chunk() -> None:
    message = build_user_message("q", [chunk("a"), chunk("b"), chunk("c")])
    assert "[c:a]" in message
    assert "[c:b]" in message
    assert "[c:c]" in message
