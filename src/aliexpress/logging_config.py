"""Shared logging configuration for the aliexpress package."""

import logging

_FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


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
        log.addHandler(handler)


def add_file_handler(logfile: str) -> None:
    """Attach a file handler to the aliexpress package logger.

    Called after the Flask instance path is known so the log file lands in
    ``instance/logs/`` rather than the project root.
    """
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)
    logging.getLogger("aliexpress").addHandler(file_handler)
