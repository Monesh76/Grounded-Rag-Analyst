"""tracing.py must be a true no-op without Langfuse keys configured -- make
test/CI's default job run with no keys set, and must never touch the
network."""

from filings_rag import tracing
from filings_rag.config import get_settings


def test_span_yields_none_without_langfuse_keys() -> None:
    settings = get_settings().model_copy(
        update={"langfuse_public_key": None, "langfuse_secret_key": None}
    )
    with tracing.span(settings, "test-span") as obs:
        assert obs is None


def test_update_on_none_is_a_no_op() -> None:
    tracing.update(None, output="anything")  # must not raise


def test_flush_without_keys_is_a_no_op() -> None:
    settings = get_settings().model_copy(
        update={"langfuse_public_key": None, "langfuse_secret_key": None}
    )
    tracing.flush(settings)  # must not raise, must not touch the network
