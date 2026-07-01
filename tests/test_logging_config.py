"""Unit tests for logging_config: LogContextEnricher and bind_log_context."""

import logging

from aliexpress.logging_config import LogContextEnricher, bind_log_context


def _make_record(msg="test") -> logging.LogRecord:
    return logging.LogRecord(
        name="aliexpress.test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_enricher_defaults_to_dashes():
    """Fields default to '-' when no context is active."""
    record = _make_record()
    LogContextEnricher().filter(record)
    assert record.school == "-"  # pylint: disable=no-member
    assert record.process == "-"  # pylint: disable=no-member
    assert record.run == "-"  # pylint: disable=no-member
    assert record.phase == "-"  # pylint: disable=no-member


def test_enricher_within_bind_log_context():
    """Fields reflect the active context inside bind_log_context."""
    record = _make_record()
    with bind_log_context(school="X", process="p", run="42", phase="solve"):
        LogContextEnricher().filter(record)
    assert record.school == "X"  # pylint: disable=no-member
    assert record.process == "p"  # pylint: disable=no-member
    assert record.run == "42"  # pylint: disable=no-member
    assert record.phase == "solve"  # pylint: disable=no-member


def test_enricher_resets_after_bind_log_context():
    """Context is restored to defaults after exiting bind_log_context."""
    with bind_log_context(school="X"):
        pass
    record = _make_record()
    LogContextEnricher().filter(record)
    assert record.school == "-"  # pylint: disable=no-member
