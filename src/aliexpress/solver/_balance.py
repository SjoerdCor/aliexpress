"""Shared primitives for the solver sub-package: ``GroupBalance`` and the solver log path."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

# Per-thread solver log path: set by _run_solve_thread before each solve so that
# concurrent processes write to separate files rather than the shared OS temp log.
_SOLVER_LOG_PATH: ContextVar[str | None] = ContextVar("_SOLVER_LOG_PATH", default=None)


@contextmanager
def solver_log_path(path: str):
    """Context manager that sets the per-thread solver log path to *path*."""
    token = _SOLVER_LOG_PATH.set(path)
    try:
        yield
    finally:
        _SOLVER_LOG_PATH.reset(token)


@dataclass
class GroupBalance:
    """
    Constraints controlling how students are distributed across groups.

    All values must be non-negative integers.
    """

    max_clique: int = 5
    """The number of students that can go to the same group"""

    max_clique_sex: int = 3
    """Maximum number of students of the same sex from the same original group in a group."""

    max_diff_n_students_year: int = 2
    """Max difference between largest and smallest group per year."""

    max_diff_n_students_total: int = 3
    """Max difference between largest and smallest group overall."""

    max_imbalance_boys_girls_year: int = 2
    """Max difference between boys and girls per year in a group."""

    max_imbalance_boys_girls_total: int = 3
    """Max difference between boys and girls in total per group."""

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


# All six balance limits set to 1: the tightest possible balance.
# Used as the starting point for the adaptive relaxation search.
STRICTEST_BALANCE = GroupBalance(1, 1, 1, 1, 1, 1)
