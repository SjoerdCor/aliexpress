"""Unit tests for ProblemSolver internal helpers."""

import math

import pandas as pd

from aliexpress.solver.problemsolver import ProblemSolver


def _make_solver(students: dict) -> ProblemSolver:
    """Minimal ProblemSolver with two target groups and no preferences."""
    preferences = pd.DataFrame(
        columns=["Waarde", "Gewicht"],
        index=pd.MultiIndex.from_tuples([], names=["Leerling", "TypeWens", "Nr"]),
    )
    groups_to = {
        "blauw": {"Jongens": 0, "Meisjes": 0},
        "rood": {"Jongens": 0, "Meisjes": 0},
    }
    return ProblemSolver(preferences, students, groups_to, [])


# ---------------------------------------------------------------------------
# _cohorts()
# ---------------------------------------------------------------------------


def test_cohorts_no_jaarlaag_gives_single_none_cohort():
    """Students without Jaarlaag fall into one implicit cohort keyed by None."""
    students = {
        "anna": {
            "Stamgroep": "A",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
        },
        "bram": {
            "Stamgroep": "A",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
        },
    }
    solver = _make_solver(students)
    cohorts = solver.cohorts()
    assert list(cohorts.keys()) == [None]
    assert set(cohorts[None]) == {"anna", "bram"}


def test_cohorts_single_jaarlaag_gives_one_cohort():
    """All students in the same Jaarlaag → one cohort (the doorzetten case)."""
    students = {
        "anna": {
            "Stamgroep": "A",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        },
        "bram": {
            "Stamgroep": "A",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        },
    }
    solver = _make_solver(students)
    cohorts = solver.cohorts()
    assert list(cohorts.keys()) == [6]
    assert set(cohorts[6]) == {"anna", "bram"}


def test_cohorts_multiple_jaarlagen_splits_correctly():
    """Multiple Jaarlaag values produce one cohort per distinct value."""
    students = {
        "anna": {
            "Stamgroep": "A",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        },
        "bram": {
            "Stamgroep": "A",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        },
        "cas": {
            "Stamgroep": "B",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 7,
        },
        "demi": {
            "Stamgroep": "B",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 7,
        },
        "eva": {
            "Stamgroep": "C",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 8,
        },
    }
    solver = _make_solver(students)
    cohorts = solver.cohorts()
    assert set(cohorts.keys()) == {6, 7, 8}
    assert set(cohorts[6]) == {"anna", "bram"}
    assert set(cohorts[7]) == {"cas", "demi"}
    assert set(cohorts[8]) == {"eva"}


def test_cohorts_mixed_present_absent_jaarlaag():
    """Students with and without Jaarlaag are separated into distinct cohorts."""
    students = {
        "anna": {
            "Stamgroep": "A",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        },
        "bram": {
            "Stamgroep": "A",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
        },
    }
    solver = _make_solver(students)
    cohorts = solver.cohorts()
    assert set(cohorts.keys()) == {6, None}
    assert cohorts[6] == ["anna"]
    assert cohorts[None] == ["bram"]
