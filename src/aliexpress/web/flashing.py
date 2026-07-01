"""Shared flash-and-log helpers for recoverable validation rejections."""

import logging

from flask import flash

logger = logging.getLogger(__name__)


def warn_and_flash(message: str, *, log_detail: str | None = None) -> None:
    """Flash *message* as an error and emit a WARNING to the package logger.

    Use this for recoverable validation rejections that the teacher sees as a
    flash message. *log_detail* should be a short English identifier (e.g. the
    exception code) that is safe to log — never include student names or other PII.
    """
    logger.warning("Validation rejected: %s", log_detail or "(unspecified)")
    flash(message, "error")
