"""Shared logging configuration for the aliexpress package."""

import logging
import logging.handlers
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("_LOG_CONTEXT", default={})

_FORMAT = (
    "%(asctime)s %(levelname)s "
    "[%(school)s/%(process)s/%(run)s %(phase)s] "
    "%(name)s %(threadName)s: %(message)s"
)
_FORMATTER = logging.Formatter(_FORMAT)


# pylint: disable=too-few-public-methods  # single-method interface imposed by logging.Filter
class LogContextEnricher(logging.Filter):
    """Enrich each LogRecord with correlation fields from the active ContextVar.

    Named a Filter because that is Python logging's hook for pre-format record
    mutation, but it never discards records — it only adds school/process/run/phase
    from the per-thread ContextVar and always returns True.
    """

    _FIELDS = ("school", "process", "run", "phase")

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _LOG_CONTEXT.get()
        for field in self._FIELDS:
            setattr(record, field, ctx.get(field, "-"))
        return True


def push_log_context(**fields):
    """Merge non-None fields into the current log context; return the reset token."""
    current = dict(_LOG_CONTEXT.get())
    current.update({k: v for k, v in fields.items() if v is not None})
    return _LOG_CONTEXT.set(current)


def pop_log_context(token) -> None:
    """Reset the log context to the state captured in *token*."""
    _LOG_CONTEXT.reset(token)


@contextmanager
def bind_log_context(**fields):
    """Context manager that sets log context fields and restores them on exit."""
    token = push_log_context(**fields)
    try:
        yield
    finally:
        pop_log_context(token)


def configure_logging() -> None:
    """Attach a DEBUG-level console handler to the aliexpress package logger.

    Call once at application startup. Idempotent: a second call is a no-op when a
    handler is already present (safe under Flask's development-mode reloader).
    All child loggers (aliexpress.*, aliexpress.app) inherit this handler via propagation.
    """
    log = logging.getLogger("aliexpress")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(_FORMATTER)
        handler.addFilter(LogContextEnricher())
        log.addHandler(handler)


def add_file_handler(logfile: str) -> None:
    """Attach a rotating file handler to the aliexpress package logger.

    Rotates at midnight, keeps 90 days of backups. Level is INFO so the file
    stays readable without the DEBUG noise that aids interactive development.
    Called after the Flask instance path is known so the log file lands in
    ``instance/logs/`` rather than the project root.
    """
    file_handler = logging.handlers.TimedRotatingFileHandler(
        logfile, when="midnight", backupCount=90, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(_FORMATTER)
    file_handler.addFilter(LogContextEnricher())
    logging.getLogger("aliexpress").addHandler(file_handler)
