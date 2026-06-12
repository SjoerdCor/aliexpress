"""Shared logging configuration for the aliexpress package."""

import logging

_FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def setup_logger(name: str) -> logging.Logger:
    """Create a DEBUG-level console logger for the given module name."""
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(_FORMATTER)
    log.addHandler(console_handler)

    return log


def add_file_handler(log: logging.Logger, logfile: str) -> None:
    """Attach a file handler to an existing logger.

    Called after the Flask instance path is known so the log file lands in
    ``instance/logs/`` rather than the project root.
    """
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)
    log.addHandler(file_handler)
