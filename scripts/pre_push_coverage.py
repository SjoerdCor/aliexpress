"""Run the resource-aware non-slow test lanes and their combined coverage gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_COVERAGE_OPTIONS = ("--cov=aliexpress", "--cov=app", "--cov-report=")
_TEST_SELECTIONS = (
    (
        "fast tests without the real solver",
        "tests",
        "--ignore=tests/integration",
        "--ignore=tests/browser",
        "-q",
        "-m",
        "not slow and not real_solver",
        "-n",
        "4",
        "--dist",
        "load",
    ),
    (
        "fast tests with the real solver",
        "tests",
        "--ignore=tests/integration",
        "--ignore=tests/browser",
        "-q",
        "-m",
        "not slow and real_solver",
        "-n",
        "0",
    ),
    (
        "browser tests without the real solver",
        "tests/browser",
        "-q",
        "-m",
        "not slow and not real_solver",
        "-n",
        "2",
        "--dist",
        "load",
    ),
    (
        "browser tests with the real solver",
        "tests/browser",
        "-q",
        "-m",
        "not slow and real_solver",
        "-n",
        "0",
    ),
    (
        "non-slow integration tests",
        "tests/integration",
        "-q",
        "-m",
        "not slow",
        "-n",
        "0",
    ),
)


def _python_module_command(module: str, *arguments: str) -> list[str]:
    """Build a command using the interpreter that runs this script."""

    return [sys.executable, "-m", module, *arguments]


def _run(command: list[str]) -> None:
    """Run one command in the project root and stop on its first failure."""

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    """Run all non-slow selections, then enforce the combined coverage threshold."""

    _run(_python_module_command("coverage", "erase"))

    for index, (description, *selection) in enumerate(_TEST_SELECTIONS):
        print(f"Running {description}...", flush=True)
        append = ("--cov-append",) if index else ()
        _run(
            _python_module_command(
                "pytest",
                *selection,
                *_COVERAGE_OPTIONS,
                *append,
            )
        )

    _run(_python_module_command("coverage", "report", "--fail-under=90"))


if __name__ == "__main__":
    main()
