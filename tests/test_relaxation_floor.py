"""Regression tests for the relaxation floor: strictly positive satisfaction.

Before ADR-0014 (negative satisfaction), the adaptive class-balance
relaxation's stage 1 minimized the number of students left without any
honored positive preference (``unmet``). That metric no longer matches what
"protected" should mean: a student with only avoid-preferences has no
positive preference to honor at all, so ``unmet`` was structurally always
true for them and never counted against the relaxation search — the old code
was happy to let them land at a deeply negative satisfaction if that
minimized balance slack elsewhere. Similarly a student with one honored
positive preference (``unmet`` = false) could still end up net negative
because an avoid-preference was violated at the same time.

The new floor instead minimizes the number of students at or below zero
satisfaction, so both scenarios below are now avoided as far as feasibility
allows.
"""

import math

import pandas as pd
import pytest

from aliexpress.solver import engine

# Two empty target groups: occupancy is entirely determined by "Niet in"
# placements below, so the balance geometry is exact and known.
_GROUPS_TO = {
    "G1": {"Jongens": 0, "Meisjes": 0},
    "G2": {"Jongens": 0, "Meisjes": 0},
}


def _student(sex, stamgroep):
    return {
        "Stamgroep": stamgroep,
        "Jongen/meisje": sex,
        "MinimaleTevredenheid": math.nan,
        "Jaarlaag": 1,
    }


def _prefs_df(records) -> pd.DataFrame:
    df = pd.DataFrame(records).set_index(["Leerling", "TypeWens", "Nr"])
    df.columns.name = "TypeWaarde"
    return df


def test_avoid_only_student_is_protected_from_negative_satisfaction():
    """A student with only an avoid-preference must not be pushed below zero.

    G1-base is {y} (1 student, fixed via "Niet in G2"), G2-base is {z, w} (2
    students, both fixed via "Niet in G1"). "x" is free and has a single
    avoid-preference against "y". Placing x in G1 gives a perfectly balanced
    2-2 split (all balance slack 0) but puts x together with the very person
    it wants to avoid, so x's satisfaction is the pure-avoid minimum (-1.0).
    Placing x in G2 gives 1-3, which costs balance slack but leaves x's
    avoid-preference honored (satisfaction +1.0).

    Under the old ``unmet``-count floor, x has no positive preference, so it is
    excluded from the requirement entirely and stage 1's optimum (0 unmet
    students) is reached with x anywhere — stage 2 then freely picks the
    balance-cheapest option (x -> G1), leaving x at -1.0. The new
    nonpositive-count floor includes x (its satisfaction can be <= 0), so
    stage 1 forces x to a strictly positive result: x -> G2.
    """
    students = {
        "x": _student("Jongen", "Sx"),
        "y": _student("Meisje", "Sy"),
        "z": _student("Jongen", "Sz"),
        "w": _student("Meisje", "Sw"),
    }
    prefs = _prefs_df(
        [
            {
                "Leerling": "x",
                "TypeWens": "Graag met",
                "Nr": 1,
                "Waarde": "y",
                "Gewicht": -1.0,
            },
            {
                "Leerling": "y",
                "TypeWens": "Niet in",
                "Nr": 1,
                "Waarde": "G2",
                "Gewicht": 1.0,
            },
            {
                "Leerling": "z",
                "TypeWens": "Niet in",
                "Nr": 1,
                "Waarde": "G1",
                "Gewicht": 1.0,
            },
            {
                "Leerling": "w",
                "TypeWens": "Niet in",
                "Nr": 1,
                "Waarde": "G1",
                "Gewicht": 1.0,
            },
        ]
    )

    solution = engine.solve_within_minimal_relaxation(
        preferences=prefs,
        students=students,
        groups_to=_GROUPS_TO,
        not_together=[],
    )

    assert solution.student_satisfaction["x"] == pytest.approx(1.0)


def test_mixed_student_with_honored_preference_but_net_negative_is_protected():
    """An honored positive preference does not exempt a student from the floor.

    G1-base is {x, p} (2 students, both fixed via "Niet in G2"), G2-base is
    {z, w, v} (3 students, all fixed via "Niet in G1"). "y" is free and has no
    preferences. Placing y in G1 gives a perfectly balanced 3-3 split (all
    balance slack 0); placing y in G2 gives 2-4, which costs balance slack.

    "x" has two preferences: a positive one for "p" (weight +1, honored since
    both are fixed to G1) and an avoid-preference against "y" (weight -2). In
    the slack-0 world (y -> G1), x ends up together with both p (preference
    honored) and y (avoid-preference violated), net weighted sum -1 ->
    pure-avoid-scaled negative satisfaction. Under the old ``unmet`` floor this
    is invisible: x has an honored positive preference, so ``unmet[x]`` is
    false and stage 1's optimum (0 unmet) is reached in the slack-0 world,
    leaving x negative. The new nonpositive-count floor looks at net
    satisfaction directly, so it forces y -> G2 instead, keeping x's
    avoid-preference honored and its satisfaction positive.
    """
    students = {
        "x": _student("Jongen", "Sx"),
        "p": _student("Meisje", "Sp"),
        "z": _student("Jongen", "Sz"),
        "w": _student("Jongen", "Sw"),
        "v": _student("Meisje", "Sv"),
        "y": _student("Meisje", "Sy"),
    }
    prefs = _prefs_df(
        [
            {
                "Leerling": "x",
                "TypeWens": "Graag met",
                "Nr": 1,
                "Waarde": "p",
                "Gewicht": 1.0,
            },
            {
                "Leerling": "x",
                "TypeWens": "Graag met",
                "Nr": 2,
                "Waarde": "y",
                "Gewicht": -2.0,
            },
            {
                "Leerling": "x",
                "TypeWens": "Niet in",
                "Nr": 1,
                "Waarde": "G2",
                "Gewicht": 1.0,
            },
            {
                "Leerling": "p",
                "TypeWens": "Niet in",
                "Nr": 1,
                "Waarde": "G2",
                "Gewicht": 1.0,
            },
            {
                "Leerling": "z",
                "TypeWens": "Niet in",
                "Nr": 1,
                "Waarde": "G1",
                "Gewicht": 1.0,
            },
            {
                "Leerling": "w",
                "TypeWens": "Niet in",
                "Nr": 1,
                "Waarde": "G1",
                "Gewicht": 1.0,
            },
            {
                "Leerling": "v",
                "TypeWens": "Niet in",
                "Nr": 1,
                "Waarde": "G1",
                "Gewicht": 1.0,
            },
        ]
    )

    solution = engine.solve_within_minimal_relaxation(
        preferences=prefs,
        students=students,
        groups_to=_GROUPS_TO,
        not_together=[],
    )

    assert solution.student_satisfaction["x"] == pytest.approx(1.0)
