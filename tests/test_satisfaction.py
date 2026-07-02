"""Tests for satisfaction.py"""

# pylint: disable=protected-access

import math

import pandas as pd
import pulp
import pytest

from aliexpress.solver import satisfaction
from aliexpress.solver.problemsolver import ProblemSolver


def test_get_satisfaction_integral_basic():
    """Integral from 0 to 1 equals 0.5."""
    assert satisfaction.get_satisfaction_integral(0, 1) == pytest.approx(0.5)


def test_get_satisfaction_integral_decreasing_importance():
    """First preference contributes more than the second."""
    val1 = satisfaction.get_satisfaction_integral(0, 1)
    val2 = satisfaction.get_satisfaction_integral(1, 2)
    assert val1 > val2


def test_get_satisfaction_percentage_basic():
    """Half the weighted preferences honored gives 0.5."""
    assert satisfaction.get_satisfaction_percentage(2.0, 4.0) == pytest.approx(0.5)


def test_get_satisfaction_percentage_zero_max():
    """Returns 1.0 when max_weight is 0 (no preferences to honor)."""
    assert satisfaction.get_satisfaction_percentage(0.0, 0.0) == pytest.approx(1.0)


def test_get_satisfaction_percentage_all_honored():
    """All preferences honored gives 1.0."""
    assert satisfaction.get_satisfaction_percentage(3.0, 3.0) == pytest.approx(1.0)


def test_powerset_and_unique_sums_hidden_helpers():
    """_all_unique_sums returns all subset sums."""
    iterable = [1, 1, 0.5, 4]
    sums = satisfaction._all_unique_sums(iterable)
    assert sums == {0, 0.5, 1, 1.5, 2, 2.5, 4, 4.5, 5, 5.5, 6, 6.5}


def test_achievable_weighted_levels_simple():
    """Correct achievable levels for a single student with two preferences."""
    df = pd.DataFrame(
        {
            "Waarde": ["A", "B"],
            "Gewicht": [1, 2],
        },
        index=pd.MultiIndex.from_product(
            [["s1", "s1"], ["Graag met"]], names=["Leerling", "TypeWens"]
        ),
    )
    result = satisfaction._achievable_weighted_levels(df)
    assert result == {0, 1, 2, 3}


def test_calculate_added_satisfaction_monotonic():
    """Satisfaction scores are positive and match the integral values."""
    df = pd.DataFrame(
        {
            "Waarde": ["A", "B"],
            "Gewicht": [1, 2],
        },
        index=pd.MultiIndex.from_product(
            [["s1", "s1"], ["Graag met"]], names=["Leerling", "TypeWens"]
        ),
    )
    added = satisfaction.calculate_added_satisfaction(df)
    assert all(isinstance(v, float) for v in added.values())
    assert all(v > 0 for v in added.values())

    expected = {1: 0.5, 2: 0.25, 3: 0.125}
    assert added == expected


# ---------------------------------------------------------------------------
# Variable bounds and big-M sizing (LP strength)
# ---------------------------------------------------------------------------


def _student() -> dict:
    return {
        "Stamgroep": "A",
        "Jongen/meisje": "Jongen",
        "MinimaleTevredenheid": math.nan,
        "Jaarlaag": 6,
    }


def _bounds_solver() -> ProblemSolver:
    """anna: one positive (w=1) and one negative (w=-1) wish; bram: one positive;
    carla: no preferences at all."""
    records = [
        ("anna", 1, "bram", 1.0),
        ("anna", 2, "carla", -1.0),
        ("bram", 1, "anna", 1.0),
    ]
    df = pd.DataFrame(
        [
            {
                "Leerling": s,
                "TypeWens": "Graag met",
                "Nr": nr,
                "Waarde": target,
                "Gewicht": w,
            }
            for s, nr, target, w in records
        ]
    ).set_index(["Leerling", "TypeWens", "Nr"])
    df.columns.name = "TypeWaarde"
    students = {name: _student() for name in ("anna", "bram", "carla")}
    groups = {
        "blauw": {"Jongens": 0, "Meisjes": 0},
        "rood": {"Jongens": 0, "Meisjes": 0},
    }
    return ProblemSolver(df, students, groups, [])


def test_student_satisfaction_variables_get_tight_bounds():
    """calculate_student_satisfaction bounds each satisfaction variable by the
    student's own achievable range, strengthening the LP relaxation."""
    solver = _bounds_solver()
    prob = pulp.LpProblem("bounds", pulp.LpMaximize)
    satisfied = solver.add_variables_which_preferences_satisfied(prob=prob)
    satisfaction.calculate_student_satisfaction(solver, satisfied, prob)

    # anna: max_satisfaction = integral(0,1) = 0.5.  Worst case: negative wish
    # violated -> integral(0,-1) = -1, normalized -1/0.5 = -2.  Best case: 1.
    anna = solver.studentsatisfaction["anna"]
    assert anna.lowBound == pytest.approx(-2.0)
    assert anna.upBound == pytest.approx(1.0)

    # bram: only a positive wish -> [0, 1].
    bram = solver.studentsatisfaction["bram"]
    assert bram.lowBound == pytest.approx(0.0)
    assert bram.upBound == pytest.approx(1.0)

    # carla: no preferences -> constant baseline 1.
    carla = solver.studentsatisfaction["carla"]
    assert carla.lowBound == pytest.approx(1.0)
    assert carla.upBound == pytest.approx(1.0)


def test_threshold_big_m_sized_from_weights():
    """The big-M for the level thresholds spans the achievable weighted range
    (max positive sum to min negative sum) instead of a fixed huge constant."""
    solver = _bounds_solver()
    graag_met = solver.preferences.droplevel("TypeWens")
    # anna: pos_sum=1, neg_sum=-1; bram: pos_sum=1 -> range [-1, 1], M = 2 + margin.
    big_m = satisfaction._threshold_big_m(graag_met)
    assert big_m == pytest.approx(3.0)
