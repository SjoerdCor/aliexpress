"""Characterisation test: SolutionResult.weighted_satisfied is derivable from satisfied + weights.

This test pins the equivalence that justifies eliminating the write-back of
self.weighted_satisfied and self.weights from _calculate_weighted_preferences.
It runs on the current code and must stay green after the write-back is removed (commit 3).
"""

from aliexpress.data.datareader import GroupCounts, matching_key
from aliexpress.data.preferences_data import PreferenceData, get_graag_met
from aliexpress.data.preferences_form import (
    Preference,
    PreferenceKind,
    StudentEntry,
    build_preference_data,
)
from aliexpress.solver.problemsolver import ProblemSolver


def _build_small_scenario() -> tuple[PreferenceData, GroupCounts]:
    """Four students, three groups, two positive and one negative preference."""
    groups_to = ["Blauw", "Geel", "Rood"]
    groups_to_keys = [matching_key(g) for g in groups_to]

    students = [
        StudentEntry(
            student="Anna",
            sex="Meisje",
            origin_group="A",
            min_satisfaction=None,
            preferences=[Preference("Bo", 1.0, PreferenceKind.TOGETHER)],
        ),
        StudentEntry(
            student="Bo",
            sex="Jongen",
            origin_group="A",
            min_satisfaction=None,
            preferences=[Preference("Anna", 1.0, PreferenceKind.TOGETHER)],
        ),
        StudentEntry(
            student="Cas",
            sex="Meisje",
            origin_group="A",
            min_satisfaction=None,
            preferences=[Preference("Daan", 1.0, PreferenceKind.APART)],
        ),
        StudentEntry(
            student="Daan",
            sex="Jongen",
            origin_group="A",
            min_satisfaction=None,
        ),
    ]

    preference_data = build_preference_data(students, groups_to_keys)
    target_groups = GroupCounts(
        counts={
            "blauw": {"Jongens": 0, "Meisjes": 0},
            "geel": {"Jongens": 0, "Meisjes": 0},
            "rood": {"Jongens": 0, "Meisjes": 0},
        },
        display={"blauw": "Blauw", "geel": "Geel", "rood": "Rood"},
    )
    return preference_data, target_groups


def test_weighted_satisfied_matches_formula():
    """result.weighted_satisfied[k] == s*w (w>0) or (1-s)*w (w<0) for all keys k."""
    preference_data, target_groups = _build_small_scenario()
    solver = ProblemSolver(
        preference_data.preferences,
        preference_data.students_info,
        target_groups.counts,
        [],
        optimize="studentsatisfaction",
    )
    solver.run()
    result = solver.extract_solution()

    graag_met = get_graag_met(preference_data.preferences)
    expected_weights = dict(graag_met["Gewicht"])

    # weights must equal the Gewicht column
    assert result.weights == expected_weights

    # weighted_satisfied must equal the analytical formula
    for key, s in result.satisfied.items():
        w = result.weights[key]
        expected = s * w if w > 0 else (1 - s) * w
        assert result.weighted_satisfied[key] == expected, (
            f"key={key}: weighted_satisfied={result.weighted_satisfied[key]}, "
            f"formula={expected} (s={s}, w={w})"
        )
