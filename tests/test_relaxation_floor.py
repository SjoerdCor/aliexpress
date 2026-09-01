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
from aliexpress.solver._balance_families import SLACK_WEIGHTS

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


# Ten students over two empty groups, all Jaarlaag 1, spread over three
# Stamgroepen (st0/st1/st2) with a lopsided boy/girl split (8 boys, 2 girls).
# Nothing forces a specific assignment (the single "Graag met" row below just
# keeps the preferences frame non-empty, an unsupported edge case otherwise) —
# the six balance families alone leave no assignment fully balanced, and two
# genuinely different relaxation vectors are reachable:
#
# - weighted sum + max-slack (the old objective): realized weighted slacks
#   sorted descending [200, 200, 100, 98, 0, 0] (clique and gender_year both
#   at 200 — the unweighted max-slack term cannot tell a weight-100 family
#   from a weight-49 one, so it is happy to pile a second family up to 200).
# - weighted leximin (the new objective): [200, 100, 100, 100, 49, 49] — the
#   same unavoidable peak (clique, 200) but only *one* family allowed to sit
#   at that peak; the second-largest weighted slack is minimized to 100
#   instead of 200.
#
# Found by randomized search over small instances (see the balansrelaxatie
# plan) rather than derived by hand; verified to fail against the pre-ADR-0018
# weighted-sum objective (second-largest weighted slack 200) and pass against
# the leximin one (second-largest 100).
_LEXIMIN_GROUPS_TO = {
    "G0": {"Jongens": 0, "Meisjes": 0},
    "G1": {"Jongens": 0, "Meisjes": 0},
}


def _leximin_students() -> dict:
    stamgroep_by_index = [
        "st0",
        "st1",
        "st2",
        "st1",
        "st2",
        "st1",
        "st1",
        "st1",
        "st1",
        "st2",
    ]
    sex_by_index = [
        "Jongen",
        "Jongen",
        "Jongen",
        "Meisje",
        "Meisje",
        "Meisje",
        "Jongen",
        "Jongen",
        "Jongen",
        "Jongen",
    ]
    return {
        f"s{i}": {
            "Stamgroep": stamgroep_by_index[i],
            "Jongen/meisje": sex_by_index[i],
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 1,
        }
        for i in range(10)
    }


def _realized_cliques(assignment: dict, students: dict, groups_to: dict) -> tuple:
    """Largest same-Stamgroep (and same-Stamgroep-same-sex) headcount in one group."""
    stamgroepen: dict = {}
    for student, info in students.items():
        stamgroepen.setdefault(info["Stamgroep"], []).append(student)
    clique = 0
    clique_sex = 0
    for members in stamgroepen.values():
        counts = {group: 0 for group in groups_to}
        counts_sex = {
            (group, sex): 0 for group in groups_to for sex in ("Jongen", "Meisje")
        }
        for student in members:
            group = assignment[student]
            counts[group] += 1
            counts_sex[group, students[student]["Jongen/meisje"]] += 1
        clique = max(clique, *counts.values())
        clique_sex = max(clique_sex, *counts_sex.values())
    return clique, clique_sex


def _realized_weighted_slacks(
    assignment: dict, students: dict, groups_to: dict
) -> dict:
    """Recompute each family's realized weighted slack from a solved ``assignment``.

    Mirrors the ``STRICTEST_LIMIT + slack`` arithmetic of
    ``_balance_families.py`` in plain Python, over the *realized* group
    counts/gender splits/clique counts — the same "derive it from the public
    assignment" style as ``_group_size_spread`` in
    ``tests/integration/test_integration_main.py``, just for all six families
    instead of one.
    """
    strictest_limit = 1

    def spread(members) -> int:
        counts = {group: 0 for group in groups_to}
        for student in members:
            counts[assignment[student]] += 1
        return max(counts.values()) - min(counts.values())

    def gender_diff_max(members) -> int:
        boys = {group: 0 for group in groups_to}
        girls = {group: 0 for group in groups_to}
        for student in members:
            group = assignment[student]
            if students[student]["Jongen/meisje"] == "Jongen":
                boys[group] += 1
            else:
                girls[group] += 1
        return max(abs(boys[group] - girls[group]) for group in groups_to)

    cohorts: dict = {}
    for student, info in students.items():
        cohorts.setdefault(info.get("Jaarlaag"), []).append(student)

    clique, clique_sex = _realized_cliques(assignment, students, groups_to)
    raw = {
        "diff_year": max(spread(members) for members in cohorts.values()),
        "diff_total": spread(list(students)),
        "clique": clique,
        "clique_sex": clique_sex,
        "gender_year": max(gender_diff_max(members) for members in cohorts.values()),
        "gender_total": gender_diff_max(list(students)),
    }
    return {
        name: SLACK_WEIGHTS[name] * max(0, value - strictest_limit)
        for name, value in raw.items()
    }


def test_balance_relaxation_prefers_lower_weighted_peak():
    """Leximin caps how many families may share the largest weighted slack.

    The old objective (weighted sum + an unweighted max-slack term) cannot
    distinguish a weight-100 family from a weight-49 one once both sit at the
    same *unweighted* slack value, so it is willing to let a second family
    join the top of the weighted slacks. Leximin instead minimizes the sorted
    weighted-slack level by level: at most one family may sit at the largest
    weighted value, so
    the second-largest is pushed down from 200 to 100 on this instance.
    """
    students = _leximin_students()
    prefs = pd.DataFrame(
        [
            {
                "Leerling": "s0",
                "TypeWens": "Graag met",
                "Nr": 1,
                "Waarde": "s1",
                "Gewicht": 1.0,
            }
        ]
    ).set_index(["Leerling", "TypeWens", "Nr"])

    solution = engine.solve_within_minimal_relaxation(
        preferences=prefs,
        students=students,
        groups_to=_LEXIMIN_GROUPS_TO,
        not_together=[],
    )

    realized = _realized_weighted_slacks(
        solution.assignment, students, _LEXIMIN_GROUPS_TO
    )
    assert sorted(realized.values(), reverse=True) == [200, 100, 100, 100, 49, 49]
