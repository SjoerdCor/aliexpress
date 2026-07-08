"""Regression test for the lexmaxmin ``minimum`` variable's bounds.

The bounds must come from the satisfaction variables' own domains, not a fixed
magic constant: a student with a mix of honored positive and violated
negative wishes can reach a satisfaction far below any small fixed lower
bound (see ``src/aliexpress/solver/strategies.py``).
"""

import math

import pandas as pd
import pytest

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
    """Five students, one target group.

    "a" has only violated negative wishes; "e" has a mix of one honored
    positive wish and violated negative wishes.
    """
    return {
        name: {
            "Stamgroep": "G1",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 1,
        }
        for name in ("a", "b", "c", "d", "e")
    }


def _build_prefs() -> pd.DataFrame:
    """Student "a" is pure-avoid; student "e" mixes an honored positive wish
    with violated negative wishes.

    Post-negation convention (see
    ``tests/integration/test_integration_herindelen.py::_build_prefs``): all
    rows are "Graag met", a negative "Gewicht" is a "Liever niet met" wish.
    With a single target group every wish is necessarily honored if positive
    and violated if negative (everyone shares the group).

    "a" has three violated negative wishes, weight -4 each: weighted sum -12,
    best 0, worst -12, so satisfaction is pure-avoid:
    F(0,-12)/|F(0,-12)| = -1.0.

    "e" has one honored positive wish (weight +1) and three violated negative
    wishes (weight -4 each): weighted sum +1-12=-11, best +1, so satisfaction
    = F(0,-11)/F(0,1) = -2047/0.5 = -4094, far below the old fixed
    ``-10 * scale`` bound on the lexmaxmin ``minimum``.
    """
    records = (
        [
            {
                "Leerling": "a",
                "TypeWens": "Graag met",
                "Nr": nr,
                "Waarde": other,
                "Gewicht": -4.0,
            }
            for nr, other in enumerate(("b", "c", "d"), start=1)
        ]
        + [
            {
                "Leerling": "e",
                "TypeWens": "Graag met",
                "Nr": 1,
                "Waarde": "a",
                "Gewicht": 1.0,
            }
        ]
        + [
            {
                "Leerling": "e",
                "TypeWens": "Graag met",
                "Nr": nr,
                "Waarde": other,
                "Gewicht": -4.0,
            }
            for nr, other in enumerate(("b", "c", "d"), start=2)
        ]
    )
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
    assert set(solution.assignment) == {"a", "b", "c", "d", "e"}
    assert all(group == "only" for group in solution.assignment.values())
    assert solution.student_satisfaction["a"] == pytest.approx(-1.0)
    assert solution.student_satisfaction["e"] == pytest.approx(-4094.0)
