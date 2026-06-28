"""Shared primitives for the solver sub-package.

Lives outside both problemsolver.py and feasibility.py to avoid a circular
import (problemsolver → feasibility → problemsolver).  Both modules import
from here; this file imports neither.
"""

import os
import tempfile
from dataclasses import dataclass

import pulp


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


def get_solver() -> pulp.HiGHS:
    """Return the HiGHS PuLP solver with proven-optimum settings."""
    # gapRel=0 so we always get the proven optimum, not an early cutoff.
    # logPath goes to the OS temp dir so HiGHS never writes into the project root.
    log_path = os.path.join(tempfile.gettempdir(), "aliexpress-solver.log")
    return pulp.HiGHS(logPath=log_path, msg=False, gapRel=0)
