"""Shared logging configuration for the aliexpress package."""

import logging


def setup_logger(name: str, logfile: str | None = None) -> logging.Logger:
    """Create a DEBUG-level logger for the given module name.

    Logs to the console, and additionally to ``logfile`` when one is given.
    """
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    if logfile is not None:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    return log
