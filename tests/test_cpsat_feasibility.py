"""Tests for the CP-SAT infeasibility diagnosis (cpsat/feasibility.py).

Each scenario is deliberately infeasible through one hard preference family;
the tests assert which family the diagnosis attributes it to. They run against
the CP-SAT model builders directly, on the plain
``(preferences, students, groups_to, not_together)`` inputs
``cpsat.model.build_feasibility_problem`` consumes.
"""

import pandas as pd
import pytest

from aliexpress import errors
from aliexpress.data.datareader import GroupCounts, matching_key
from aliexpress.data.preferences_form import (
    Preference,
    PreferenceKind,
    StudentEntry,
    build_preference_data,
)
from aliexpress.solver.cpsat import engine, feasibility


def _target_groups(*names: str) -> GroupCounts:
    """A ``GroupCounts`` with all-empty occupancy for the given group names."""
    keys = [matching_key(name) for name in names]
    return GroupCounts(
        counts={key: {"Jongens": 0, "Meisjes": 0} for key in keys},
        display=dict(zip(keys, names)),
    )


def _infeasible_by_min_satisfaction():
    """Anna must be with Bo (min_satisfaction=1.0) but excluded groups force them apart."""
    groups_to = ["Blauw", "Geel", "Rood"]
    groups_to_keys = [matching_key(g) for g in groups_to]

    students = [
        StudentEntry(
            student="Anna",
            sex="Meisje",
            origin_group="A",
            min_satisfaction=1.0,
            preferences=[Preference("Bo", 1.0, PreferenceKind.TOGETHER)],
            excluded_groups=["Rood"],
        ),
        StudentEntry(
            student="Bo",
            sex="Jongen",
            origin_group="A",
            min_satisfaction=None,
            excluded_groups=["Blauw", "Geel"],
        ),
        StudentEntry("Cas", "Meisje", "A", None),
        StudentEntry("Daan", "Jongen", "A", None),
        StudentEntry("Eva", "Meisje", "A", None),
        StudentEntry("Finn", "Jongen", "A", None),
    ]

    preference_data = build_preference_data(students, groups_to_keys)
    target_groups = _target_groups(*groups_to)
    return preference_data, target_groups


def _infeasible_by_not_together():
    """Ali/Bram/Cis can only go to Rood, but a not-together rule allows at most one."""
    groups_to = ["Blauw", "Geel", "Rood"]
    groups_to_keys = [matching_key(g) for g in groups_to]
    only_rood = ["Blauw", "Geel"]

    students = [
        StudentEntry("Ali", "Jongen", "A", None, excluded_groups=only_rood),
        StudentEntry("Bram", "Jongen", "A", None, excluded_groups=only_rood),
        StudentEntry("Cis", "Meisje", "A", None, excluded_groups=only_rood),
        StudentEntry("Daan", "Jongen", "A", None),
        StudentEntry("Eva", "Meisje", "A", None),
        StudentEntry("Finn", "Jongen", "A", None),
    ]

    preference_data = build_preference_data(students, groups_to_keys)
    target_groups = _target_groups(*groups_to)
    not_together = [{"group": {"ali", "bram", "cis"}, "Max_aantal_samen": 1}]
    return preference_data, target_groups, not_together


def _infeasible_fundamental():
    """A student excluded from every group: no relaxation of either family can help.

    Built as a raw preferences frame (not via ``build_preference_data``, which
    rejects an ``excluded_groups`` set covering every destination group) with
    three "Niet in" rows for the same student against all three groups.
    """
    groups_to = ["blauw", "geel", "rood"]
    records = [
        {
            "Leerling": "anna",
            "TypeWens": "Niet in",
            "Nr": float(i + 1),
            "Waarde": group,
            "Gewicht": 1.0,
        }
        for i, group in enumerate(groups_to)
    ]
    preferences = pd.DataFrame(records).set_index(["Leerling", "TypeWens", "Nr"])
    students = {
        "anna": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Meisje",
            "Stamgroep": "a",
        },
        "bo": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Jongen",
            "Stamgroep": "a",
        },
    }
    groups_to_dict = {group: {"Jongens": 0, "Meisjes": 0} for group in groups_to}
    return preferences, students, groups_to_dict


class TestFeasibleWhenRelaxed:
    """feasible_when_relaxed: leave-one-out feasibility checks for each family."""

    def test_min_satisfaction_infeasible_relaxed_min_sat_is_feasible(self):
        """Relaxing min_satisfaction resolves infeasibility caused by that family."""
        preference_data, target_groups = _infeasible_by_min_satisfaction()
        assert feasibility.feasible_when_relaxed(
            preferences=preference_data.preferences,
            students=preference_data.students_info,
            groups_to=target_groups.counts,
            not_together=[],
            min_satisfaction_soft=True,
            not_together_soft=False,
        )

    def test_min_satisfaction_infeasible_relaxed_not_together_stays_infeasible(self):
        """Relaxing not_together alone does not resolve min_satisfaction infeasibility."""
        preference_data, target_groups = _infeasible_by_min_satisfaction()
        assert not feasibility.feasible_when_relaxed(
            preferences=preference_data.preferences,
            students=preference_data.students_info,
            groups_to=target_groups.counts,
            not_together=[],
            min_satisfaction_soft=False,
            not_together_soft=True,
        )

    def test_not_together_infeasible_relaxed_not_together_is_feasible(self):
        """Relaxing not_together resolves infeasibility caused by that family."""
        preference_data, target_groups, not_together = _infeasible_by_not_together()
        assert feasibility.feasible_when_relaxed(
            preferences=preference_data.preferences,
            students=preference_data.students_info,
            groups_to=target_groups.counts,
            not_together=not_together,
            min_satisfaction_soft=False,
            not_together_soft=True,
        )

    def test_not_together_infeasible_relaxed_min_sat_stays_infeasible(self):
        """Relaxing min_satisfaction alone does not resolve not_together infeasibility."""
        preference_data, target_groups, not_together = _infeasible_by_not_together()
        assert not feasibility.feasible_when_relaxed(
            preferences=preference_data.preferences,
            students=preference_data.students_info,
            groups_to=target_groups.counts,
            not_together=not_together,
            min_satisfaction_soft=True,
            not_together_soft=False,
        )


class TestDiagnose:
    """diagnose: family attribution for infeasible preference configurations."""

    def test_min_satisfaction_case(self):
        """Reports min_satisfaction when that family alone causes infeasibility."""
        preference_data, target_groups = _infeasible_by_min_satisfaction()
        assert (
            feasibility.diagnose(
                preferences=preference_data.preferences,
                students=preference_data.students_info,
                groups_to=target_groups.counts,
                not_together=[],
            )
            == "min_satisfaction"
        )

    def test_not_together_case(self):
        """Reports not_together when that family alone causes infeasibility."""
        preference_data, target_groups, not_together = _infeasible_by_not_together()
        assert (
            feasibility.diagnose(
                preferences=preference_data.preferences,
                students=preference_data.students_info,
                groups_to=target_groups.counts,
                not_together=not_together,
            )
            == "not_together"
        )

    def test_fundamental_case(self):
        """Reports fundamental when a "Niet in" exclusion is the real cause."""
        preferences, students, groups_to = _infeasible_fundamental()
        assert (
            feasibility.diagnose(
                preferences=preferences,
                students=students,
                groups_to=groups_to,
                not_together=[],
            )
            == "fundamental"
        )


def test_infeasible_auto_path_raises_diagnosed_error():
    """solve_within_minimal_relaxation raises FeasibilityError with the diagnosed case."""
    preference_data, target_groups = _infeasible_by_min_satisfaction()
    with pytest.raises(errors.FeasibilityError) as exc_info:
        engine.solve_within_minimal_relaxation(
            preferences=preference_data.preferences,
            students=preference_data.students_info,
            groups_to=target_groups.counts,
            not_together=[],
        )
    assert exc_info.value.code == "infeasible_preferences"
    assert exc_info.value.context["case"] == "min_satisfaction"
