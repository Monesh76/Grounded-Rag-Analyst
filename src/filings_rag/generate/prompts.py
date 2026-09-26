"""Builds the system prompt and the user message for question answering."""

from filings_rag.retrieve.models import RetrievalResult


def build_system_prompt(refusal_text: str) -> str:
    return (
        "You are a financial research assistant answering questions about companies' "
        "SEC 10-K filings using only the excerpts provided below.\n\n"
        "Rules:\n"
        "- Answer only using the provided excerpts. Do not use any outside knowledge, "
        "even if you are confident it is correct.\n"
        "- Cite every factual claim with the excerpt's id in this exact format: [c:<id>]. "
        "Place the citation immediately after the sentence it supports.\n"
        "- Cite only ids that appear in the excerpts below. Never invent or guess an id.\n"
        "- If the excerpts do not contain enough information to answer the question, "
        f'reply with exactly this sentence and nothing else: "{refusal_text}"\n'
    )


def build_user_message(question: str, chunks: list[RetrievalResult]) -> str:
    excerpts = "\n\n".join(
        f"[c:{c.id}] ({c.company} FY{c.fiscal_year}, Item {c.item} - {c.section_title}, "
        f"p.{c.page}):\n{c.text}"
        for c in chunks
    )
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"
