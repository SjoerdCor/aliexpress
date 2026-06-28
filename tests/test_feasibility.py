"""Characterisation tests for feasibility.py — diagnose and feasible_when_relaxed."""

import pytest

from aliexpress import errors, feasibility
from aliexpress.datareader import GroupCounts, matching_key
from aliexpress.main import distribute_students_from_data
from aliexpress.preferences_form import (
    Preference,
    PreferenceKind,
    StudentEntry,
    build_preference_data,
)
from aliexpress.problemsolver import ProblemSolver
from aliexpress.validation_messages import to_validation_message


def _make_solver(preference_data, target_groups, not_together=None):
    """Build a ProblemSolver from high-level fixture data, without solving."""
    return ProblemSolver(
        preference_data.preferences,
        preference_data.students_info,
        target_groups.counts,
        not_together or [],
        optimize="lexmaxmin",
    )


def _infeasible_by_min_satisfaction():
    """Fixture: Anna must be with Bo (min_satisfaction=1.0) but excluded groups force them apart."""
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
    target_groups = GroupCounts(
        counts={
            "blauw": {"Jongens": 0, "Meisjes": 0},
            "geel": {"Jongens": 0, "Meisjes": 0},
            "rood": {"Jongens": 0, "Meisjes": 0},
        },
        display={"blauw": "Blauw", "geel": "Geel", "rood": "Rood"},
    )
    return preference_data, target_groups


def _infeasible_by_not_together():
    """Fixture: Ali/Bram/Cis can only go to Rood but a not-together rule allows at most one."""
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
    target_groups = GroupCounts(
        counts={
            "blauw": {"Jongens": 0, "Meisjes": 0},
            "geel": {"Jongens": 0, "Meisjes": 0},
            "rood": {"Jongens": 0, "Meisjes": 0},
        },
        display={"blauw": "Blauw", "geel": "Geel", "rood": "Rood"},
    )
    not_together = [{"group": {"ali", "bram", "cis"}, "Max_aantal_samen": 1}]
    return preference_data, target_groups, not_together


class TestFeasibleWhenRelaxed:
    """feasibility.feasible_when_relaxed: leave-one-out feasibility checks."""

    def test_min_satisfaction_infeasible_relaxed_min_sat_is_feasible(self):
        """Relaxing min_satisfaction resolves infeasibility caused by that family."""
        preference_data, target_groups = _infeasible_by_min_satisfaction()
        solver = _make_solver(preference_data, target_groups)
        assert feasibility.feasible_when_relaxed(
            solver, min_satisfaction_soft=True, not_together_soft=False
        )

    def test_min_satisfaction_infeasible_relaxed_not_together_stays_infeasible(self):
        """Relaxing not_together alone does not resolve min_satisfaction infeasibility."""
        preference_data, target_groups = _infeasible_by_min_satisfaction()
        solver = _make_solver(preference_data, target_groups)
        assert not feasibility.feasible_when_relaxed(
            solver, min_satisfaction_soft=False, not_together_soft=True
        )

    def test_not_together_infeasible_relaxed_not_together_is_feasible(self):
        """Relaxing not_together resolves infeasibility caused by that family."""
        preference_data, target_groups, not_together = _infeasible_by_not_together()
        solver = _make_solver(preference_data, target_groups, not_together)
        assert feasibility.feasible_when_relaxed(
            solver, min_satisfaction_soft=False, not_together_soft=True
        )

    def test_not_together_infeasible_relaxed_min_sat_stays_infeasible(self):
        """Relaxing min_satisfaction alone does not resolve not_together infeasibility."""
        preference_data, target_groups, not_together = _infeasible_by_not_together()
        solver = _make_solver(preference_data, target_groups, not_together)
        assert not feasibility.feasible_when_relaxed(
            solver, min_satisfaction_soft=True, not_together_soft=False
        )


class TestDiagnose:
    """feasibility.diagnose: family attribution for infeasible preference configurations."""

    def test_min_satisfaction_case(self):
        """Reports min_satisfaction when that family alone causes infeasibility."""
        preference_data, target_groups = _infeasible_by_min_satisfaction()
        solver = _make_solver(preference_data, target_groups)
        assert feasibility.diagnose(solver) == "min_satisfaction"

    def test_not_together_case(self):
        """Reports not_together when that family alone causes infeasibility."""
        preference_data, target_groups, not_together = _infeasible_by_not_together()
        solver = _make_solver(preference_data, target_groups, not_together)
        assert feasibility.diagnose(solver) == "not_together"


class TestAutoPathIntegration:
    """End-to-end: distribute_students_from_data raises and diagnoses correctly."""

    def test_infeasible_auto_path_raises_friendly_error(self):
        """Infeasible preferences raise a FeasibilityError that renders in Dutch."""
        preference_data, target_groups = _infeasible_by_min_satisfaction()
        with pytest.raises(errors.FeasibilityError) as exc_info:
            distribute_students_from_data(
                preference_data, target_groups, groupbalance=None
            )
        assert exc_info.value.code == "infeasible_preferences"

    def test_min_satisfaction_family_is_reported(self):
        """Infeasibility from minimal satisfaction is diagnosed and reported in Dutch."""
        preference_data, target_groups = _infeasible_by_min_satisfaction()
        with pytest.raises(errors.FeasibilityError) as exc_info:
            distribute_students_from_data(
                preference_data, target_groups, groupbalance=None
            )
        assert exc_info.value.context["case"] == "min_satisfaction"
        assert "extra zekerheid" in to_validation_message(exc_info.value).lower()

    def test_not_together_family_is_reported(self):
        """Infeasibility from not-together rules is diagnosed and reported in Dutch."""
        preference_data, target_groups, not_together = _infeasible_by_not_together()
        with pytest.raises(errors.FeasibilityError) as exc_info:
            distribute_students_from_data(
                preference_data,
                target_groups,
                not_together=not_together,
                groupbalance=None,
            )
        assert exc_info.value.context["case"] == "not_together"
        assert "niet-samen" in to_validation_message(exc_info.value).lower()
