from filings_rag.retrieve.filters import extract_filters


def test_extracts_ticker_from_company_name() -> None:
    assert extract_filters("What was Apple's revenue in fiscal 2024?").ticker == "AAPL"


def test_extracts_ticker_from_ticker_symbol() -> None:
    assert extract_filters("What risks does NVDA disclose?").ticker == "NVDA"


def test_extracts_year() -> None:
    assert extract_filters("What was Visa's net revenue in fiscal 2025?").fiscal_year == 2025


def test_prefers_longer_company_name_match() -> None:
    # "Bank of America" must win outright, not a partial/wrong match.
    assert extract_filters("How did Bank of America perform in 2023?").ticker == "BAC"


def test_no_company_or_year_returns_none_for_both() -> None:
    filters = extract_filters("What supply-chain risks are commonly disclosed?")
    assert filters.ticker is None
    assert filters.fiscal_year is None


def test_company_name_match_is_case_insensitive() -> None:
    assert extract_filters("what did apple report").ticker == "AAPL"


def test_does_not_match_company_name_as_a_substring_of_another_word() -> None:
    # "Visa" must not match inside an unrelated word.
    assert extract_filters("Did the company revisa its guidance?").ticker is None
