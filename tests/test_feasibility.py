"""Characterisation tests for feasibility.py — diagnose and feasible_when_relaxed."""

import pytest

from aliexpress import errors
from aliexpress.data import datareader
from aliexpress.data.datareader import GroupCounts, matching_key
from aliexpress.data.preferences_data import PreferenceData
from aliexpress.data.preferences_form import (
    Preference,
    PreferenceKind,
    StudentEntry,
    build_preference_data,
)
from aliexpress.main import distribute_students_from_data
from aliexpress.solver import feasibility
from aliexpress.solver.problemsolver import ProblemSolver
from aliexpress.web.validation_messages import to_validation_message

_FULL_INTEGRATION_DIR = "tests/integration"


def _make_solver(preference_data, target_groups, not_together=None):
    """Build a ProblemSolver from high-level fixture data, without solving."""
    return ProblemSolver(
        preference_data.preferences,
        preference_data.students_info,
        target_groups.counts,
        not_together or [],
        optimize="lexmaxmin",
    )


def _load_full_scenario():
    """Load the full integration scenario (44 students, 4 target groups)."""
    with open(f"{_FULL_INTEGRATION_DIR}/voorkeuren_full.json", encoding="utf-8") as fh:
        preference_data = PreferenceData.from_json(fh.read())
    target_groups = datareader.read_groups_excel(
        f"{_FULL_INTEGRATION_DIR}/groepen.xlsx"
    )
    return preference_data, target_groups


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


def _balanced_feasible():
    """Fixture: 6 students spread evenly across 3 origin groups, no preferences.

    With GroupBalance(1,1,1,1,1,1) (the strictest base used during the automatic solve)
    every balance constraint is exactly satisfiable without any relaxation, so R* = 0.0.
    Two students per origin group (1 boy + 1 girl) fill 3 equally-sized target groups
    with one from each origin group — all limits stay at or below 1.
    """
    groups_to_keys = [matching_key(g) for g in ["Blauw", "Geel", "Rood"]]
    students = [
        StudentEntry("Anna", "Meisje", "A", None),
        StudentEntry("Bo", "Jongen", "A", None),
        StudentEntry("Cas", "Meisje", "B", None),
        StudentEntry("Daan", "Jongen", "B", None),
        StudentEntry("Eva", "Meisje", "C", None),
        StudentEntry("Finn", "Jongen", "C", None),
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


class TestMinimalRelaxationBudget:
    """feasibility.minimal_relaxation_budget: R* computation."""

    def test_budget_is_zero_for_perfectly_balanced_input(self):
        """Perfectly balanced input needs no relaxation: R* = 0.0."""
        preference_data, target_groups = _balanced_feasible()
        solver = _make_solver(preference_data, target_groups)
        assert feasibility.minimal_relaxation_budget(solver) == pytest.approx(0.0)

    def test_budget_is_nonzero_for_full_scenario(self):
        """Full integration scenario (44 students, 4 groups) yields R* = 8.98.

        Pinned empirically: this value is the unique, solver-independent minimum
        balance relaxation under which every student can still reach a positive wish.
        A regression here means the budget logic changed behaviour.
        """
        preference_data, target_groups = _load_full_scenario()
        solver = _make_solver(preference_data, target_groups)
        assert feasibility.minimal_relaxation_budget(solver) == pytest.approx(8.98)

    def test_budget_raises_feasibility_error_for_infeasible_preferences(self):
        """Hard preference clash raises FeasibilityError('infeasible_preferences')."""
        preference_data, target_groups = _infeasible_by_min_satisfaction()
        solver = _make_solver(preference_data, target_groups)
        with pytest.raises(errors.FeasibilityError) as exc_info:
            feasibility.minimal_relaxation_budget(solver)
        assert exc_info.value.code == "infeasible_preferences"


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
