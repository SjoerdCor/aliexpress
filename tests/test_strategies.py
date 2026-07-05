"""Regression test for the lexmaxmin ``minimum`` variable's bounds.

The bounds must come from the satisfaction variables' own domains, not a fixed
magic constant: a student with only violated negative wishes can reach a
satisfaction far below any small fixed lower bound (see
``src/aliexpress/solver/strategies.py``).
"""

import math

import pandas as pd

from aliexpress.solver import engine
from aliexpress.solver._balance import GroupBalance

# A single target group: every student is forced into it, so the assignment
# is fixed regardless of the balance limits.
_GROUPS_TO = {"only": {"Jongens": 0, "Meisjes": 0}}

# Generous enough that the single-group forced assignment never trips it.
_BALANCE = GroupBalance(
    max_clique=10,
    max_clique_sex=10,
    max_diff_n_students_year=10,
    max_diff_n_students_total=10,
    max_imbalance_boys_girls_year=10,
    max_imbalance_boys_girls_total=10,
)


def _build_students():
    """Four students, one target group, no positive wishes anywhere."""
    return {
        name: {
            "Stamgroep": "G1",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 1,
        }
        for name in ("a", "b", "c", "d")
    }


def _build_prefs() -> pd.DataFrame:
    """Student "a" has three violated negative wishes, weight -4 each.

    Post-negation convention (see
    ``tests/integration/test_integration_herindelen.py::_build_prefs``): all
    rows are "Graag met", a negative "Gewicht" is a "Liever niet met" wish.
    With a single target group every wish is necessarily violated, so "a"'s
    weighted sum is -12: satisfaction = 1 + F(0, -12) = 1 - 4095 = -4094, far
    below the old fixed ``-10 * scale`` bound on the lexmaxmin ``minimum``.
    """
    records = [
        {
            "Leerling": "a",
            "TypeWens": "Graag met",
            "Nr": nr,
            "Waarde": other,
            "Gewicht": -4.0,
        }
        for nr, other in enumerate(("b", "c", "d"), start=1)
    ]
    df = pd.DataFrame(records).set_index(["Leerling", "TypeWens", "Nr"])
    df.columns.name = "TypeWaarde"
    return df


def test_lexmaxmin_handles_satisfaction_far_below_fixed_bound():
    """A feasible instance whose optimum lies below the old ``-10 * scale``
    bound must still solve, not be reported infeasible."""
    solution = engine.solve_with_fixed_balance(
        preferences=_build_prefs(),
        students=_build_students(),
        groups_to=_GROUPS_TO,
        not_together=[],
        groupbalance=_BALANCE,
    )
    assert set(solution.assignment) == {"a", "b", "c", "d"}
    assert all(group == "only" for group in solution.assignment.values())
