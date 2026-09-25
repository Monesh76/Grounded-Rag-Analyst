"""The 10 companies in scope (PLAN.md section 4)."""

from filings_rag.ingest.models import Company

COMPANIES: list[Company] = [
    # Tech
    Company(ticker="AAPL", cik=320193, name="Apple"),
    Company(ticker="MSFT", cik=789019, name="Microsoft"),
    Company(ticker="GOOGL", cik=1652044, name="Alphabet"),
    Company(ticker="AMZN", cik=1018724, name="Amazon"),
    Company(ticker="NVDA", cik=1045810, name="NVIDIA"),
    # Financials
    Company(ticker="JPM", cik=19617, name="JPMorgan Chase"),
    Company(ticker="BAC", cik=70858, name="Bank of America"),
    Company(ticker="GS", cik=886982, name="Goldman Sachs"),
    Company(ticker="V", cik=1403161, name="Visa"),
    Company(ticker="AXP", cik=4962, name="American Express"),
]
