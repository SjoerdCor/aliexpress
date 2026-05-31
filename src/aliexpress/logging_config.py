"""Shared logging configuration for the aliexpress package."""

import logging


def setup_logger(name: str) -> logging.Logger:
    """Create a console logger at DEBUG level for the given module name."""
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    return log
