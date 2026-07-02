"""Shared primitives for the solver sub-package.

Lives outside both problemsolver.py and feasibility.py to avoid a circular
import (problemsolver → feasibility → problemsolver).  Both modules import
from here; this file imports neither.
"""

import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import highspy
import pulp

# Per-thread solver log path: set by _run_solve_thread before each solve so that
# concurrent processes write to separate files rather than the shared OS temp log.
_SOLVER_LOG_PATH: ContextVar[str | None] = ContextVar("_SOLVER_LOG_PATH", default=None)


@contextmanager
def solver_log_path(path: str):
    """Context manager that routes the HiGHS solver log to *path* for this thread."""
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


def _initial_solution(lp: pulp.LpProblem) -> list | None:
    """Current variable values as a HiGHS start vector, or None when incomplete.

    The vector follows ``lp.variables()`` order, which is the column order used by
    pulp's HiGHS ``buildSolverModel``.  Returns None when any variable has no value
    yet (nothing to warm-start from, e.g. the very first solve).
    """
    values = [v.varValue for v in lp.variables()]
    if any(value is None for value in values):
        return None
    return values


class WarmStartHiGHS(pulp.HiGHS):
    """HiGHS solver that passes the variables' current values as a MIP start.

    The sequential solves (lexmaxmin levels, the lexicographic budget stages) each
    re-solve a grown version of the same problem, and the previous optimum — plus
    initial values the algorithm sets for its new level variables — is a feasible
    start.  Handing it to HiGHS gives branch-and-bound an immediate incumbent to
    prune with, instead of spending most of its time finding a first good solution.
    An infeasible or incomplete start is simply ignored by HiGHS, so this is safe
    for every solve.
    """

    def callSolver(self, lp):
        start = _initial_solution(lp)
        if start is not None:
            solution = highspy.HighsSolution()
            solution.col_value = start
            lp.solverModel.setSolution(solution)
        super().callSolver(lp)


def get_solver() -> pulp.HiGHS:
    """Return the HiGHS PuLP solver with proven-optimum settings."""
    # gapRel=0 so we always get the proven optimum, not an early cutoff.
    # logPath: use the per-thread path when set (isolates concurrent processes),
    # else fall back to the shared OS temp path.
    log_path = _SOLVER_LOG_PATH.get() or os.path.join(
        tempfile.gettempdir(), "aliexpress-solver.log"
    )
    return WarmStartHiGHS(logPath=log_path, msg=False, gapRel=0)
