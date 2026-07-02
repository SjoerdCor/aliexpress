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

import numpy as np
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


def _partial_start(lp: pulp.LpProblem) -> tuple | None:
    """Indices and values of the variables that already carry a value.

    Returns ``(indices, values)`` numpy arrays over ``var.index`` (the column
    numbers assigned by pulp's HiGHS ``buildSolverModel``), or None when no
    variable has a value yet (nothing to warm-start from: the very first solve).
    """
    valued = [(v.index, v.varValue) for v in lp.variables() if v.varValue is not None]
    if not valued:
        return None
    indices, values = zip(*valued)
    return np.array(indices, dtype=np.int32), np.array(values, dtype=np.float64)


class WarmStartHiGHS(pulp.HiGHS):
    """HiGHS solver that passes the variables' current values as a MIP start.

    The sequential solves (lexmaxmin levels, the lexicographic budget stages) each
    re-solve a grown or derived version of the same problem, and the previous
    solution over the *shared* variables is a high-quality start.  The values are
    handed to HiGHS as a partial MIP start: HiGHS completes the unvalued variables
    itself by solving a small restricted MIP.  This gives branch-and-bound an
    immediate incumbent to prune with, instead of spending most of its time finding
    a first good solution.  An infeasible start is simply discarded by HiGHS, so
    this is safe for every solve.

    ``threads`` defaults to every logical core here, in the class itself, because
    HiGHS initializes its global scheduler at the first parallel solve in the
    process: a later solve asking for *more* threads than that first one fails with
    'Not Solved'.  A class-level default makes the first solve the largest ask, so
    the order of solves can never break (verified empirically).
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("threads", os.cpu_count())
        super().__init__(*args, **kwargs)

    def callSolver(self, lp):
        start = _partial_start(lp)
        if start is not None:
            indices, values = start
            lp.solverModel.setSolution(len(indices), indices, values)
        super().callSolver(lp)


def get_solver() -> pulp.HiGHS:
    """Return the HiGHS PuLP solver with proven-optimum settings."""
    # gapRel=0 so we always get the proven optimum, not an early cutoff.
    # logPath: use the per-thread path when set (isolates concurrent processes),
    # else fall back to the shared OS temp path.
    log_path = _SOLVER_LOG_PATH.get() or os.path.join(
        tempfile.gettempdir(), "aliexpress-solver.log"
    )
    return WarmStartHiGHS(
        logPath=log_path,
        msg=False,
        gapRel=0,
        # Parallel tree search comes from the WarmStartHiGHS class default (all
        # logical cores; measured: 8 threads closed a stage-1 gap from 99.9% to 4.7%
        # within the same time budget).
        #
        # Spend more of the search on finding incumbents (HiGHS default: 0.05).  Since
        # the big-M tightening the bound side is strong; the measured bottleneck is the
        # primal side (improving warm-start incumbents), which heuristics accelerate.
        # The optimum is unaffected - only the order of exploration changes.
        mip_heuristic_effort=0.25,
    )
