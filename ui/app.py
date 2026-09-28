"""Streamlit UI for FilingsRAG (`streamlit run ui/app.py`).

A thin HTTP client over the FastAPI /ask endpoint -- no filings_rag import
beyond the fixed company list for the filter dropdown, so this can be
deployed and scaled independently of the API (P7's cheap-hosting path runs
this on Streamlit Community Cloud, the API on Cloud Run).
"""

import os

import requests
import streamlit as st

from filings_rag.ingest.companies import COMPANIES

API_BASE_URL = os.environ.get("FILINGS_RAG_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 60

st.set_page_config(page_title="FilingsRAG", page_icon="📄", layout="centered")

st.title("FilingsRAG")
st.caption("Question answering over SEC 10-K filings, with grounded citations.")

with st.sidebar:
    st.header("Filters")
    st.caption("Optional -- narrows retrieval to one company and/or fiscal year.")
    company_options = ["Any"] + [f"{c.name} ({c.ticker})" for c in COMPANIES]
    company_choice = st.selectbox("Company", company_options)
    year_choice = st.selectbox("Fiscal year", ["Any", 2023, 2024, 2025])

question = st.text_input(
    "Ask a question about these companies' 10-K filings",
    placeholder="What was Apple's total net sales in fiscal 2025?",
)
ask_clicked = st.button("Ask", type="primary")


def _selected_ticker() -> str | None:
    if company_choice == "Any":
        return None
    return company_choice.rsplit("(", 1)[1].rstrip(")")


def _render_sources(sources: list[dict]) -> None:
    st.subheader("Sources")
    for source in sources:
        label = (
            f"[c:{source['id']}] {source['company']} FY{source['fiscal_year']} "
            f"Item {source['item']} -- {source['section_title']} (p.{source['page']})"
        )
        with st.expander(label):
            st.text(source["text"])


if ask_clicked:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        payload = {
            "question": question,
            "ticker": _selected_ticker(),
            "fiscal_year": None if year_choice == "Any" else year_choice,
        }
        try:
            with st.spinner("Retrieving and generating..."):
                response = requests.post(
                    f"{API_BASE_URL}/ask", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
                )
            response.raise_for_status()
            answer = response.json()
        except requests.exceptions.RequestException as exc:
            st.error(f"Couldn't reach the API at {API_BASE_URL}: {exc}")
        else:
            if not answer["grounded"] and not answer["sources"]:
                st.info(answer["answer"])
            else:
                st.markdown(answer["answer"])
                if answer["sources"]:
                    _render_sources(answer["sources"])
            st.caption(
                f"{answer['latency_ms']:.0f} ms | {answer['tokens']} tokens | "
                f"${answer['cost_usd']:.4f}"
            )
