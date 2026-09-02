"""Tests for the CP-SAT infeasibility diagnosis (solver/feasibility.py).

Each scenario is deliberately infeasible through one hard preference family;
the tests assert which family the diagnosis attributes it to. They run against
the CP-SAT model builders directly, on the plain
``(preferences, students, groups_to, not_together)`` inputs
``model.build_feasibility_problem`` consumes.
"""

from unittest.mock import MagicMock

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
from aliexpress.solver import engine, feasibility
from aliexpress.solver._balance import BalanceMaxima


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


def test_balance_cap_diagnosis_suggests_minimal_clique_increase():
    """A clique cap of one is diagnosed as the sole cause of infeasibility."""
    students = {
        f"s{i}": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Jongen" if i < 2 else "Meisje",
            "Stamgroep": "a",
            "Jaarlaag": 1,
        }
        for i in range(5)
    }
    groups_to = {
        group: {"Jongens": 0, "Meisjes": 0} for group in ("rood", "geel", "blauw")
    }
    preferences = pd.DataFrame(
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
    capped = BalanceMaxima(max_clique=1)

    unrestricted = engine.solve_within_minimal_relaxation(
        preferences=preferences,
        students=students,
        groups_to=groups_to,
        not_together=[],
    )
    suggestion = feasibility.diagnose_balance_caps(
        preferences=preferences,
        students=students,
        groups_to=groups_to,
        not_together=[],
        maxima=capped,
    )
    relaxed = engine.solve_within_minimal_relaxation(
        preferences=preferences,
        students=students,
        groups_to=groups_to,
        not_together=[],
        maxima=BalanceMaxima(max_clique=2),
    )

    assert unrestricted.assignment
    assert suggestion == {"clique": {"current": 1, "suggested": 2}}
    assert relaxed.assignment


def test_capped_floor_infeasibility_raises_actionable_balance_error():
    """The automatic solver exposes the cap diagnosis in its public error."""
    students = {
        f"s{i}": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Jongen" if i < 2 else "Meisje",
            "Stamgroep": "a",
            "Jaarlaag": 1,
        }
        for i in range(5)
    }
    groups_to = {
        group: {"Jongens": 0, "Meisjes": 0} for group in ("rood", "geel", "blauw")
    }
    preferences = pd.DataFrame(
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

    with pytest.raises(errors.FeasibilityError) as exc_info:
        engine.solve_within_minimal_relaxation(
            preferences=preferences,
            students=students,
            groups_to=groups_to,
            not_together=[],
            maxima=BalanceMaxima(max_clique=1),
        )

    assert exc_info.value.code == "balance_caps_too_tight"
    assert exc_info.value.context == {
        "suggestion": {"clique": {"current": 1, "suggested": 2}}
    }


def test_balance_cap_diagnosis_emits_no_progress_events():
    """The silent cap diagnosis does not add UI stages or interim results."""
    students = {
        f"s{i}": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Jongen" if i < 2 else "Meisje",
            "Stamgroep": "a",
            "Jaarlaag": 1,
        }
        for i in range(5)
    }
    groups_to = {
        group: {"Jongens": 0, "Meisjes": 0} for group in ("rood", "geel", "blauw")
    }
    preferences = pd.DataFrame(
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
    listener = MagicMock()

    with pytest.raises(errors.FeasibilityError):
        engine.solve_within_minimal_relaxation(
            preferences=preferences,
            students=students,
            groups_to=groups_to,
            not_together=[],
            maxima=BalanceMaxima(max_clique=1),
            listener=listener,
        )

    listener.stage_started.assert_called_once_with("floor")
    listener.stage_finished.assert_not_called()
    listener.interim_result.assert_not_called()


def test_balance_cap_diagnosis_returns_joint_multiple_overflows():
    """All positive cap overflows are returned as one jointly feasible set."""
    students = {
        f"s{i}": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Jongen" if i < 2 else "Meisje",
            "Stamgroep": "a",
            "Jaarlaag": 1,
        }
        for i in range(5)
    }
    groups_to = {group: {"Jongens": 0, "Meisjes": 0} for group in ("rood", "geel")}
    preferences = pd.DataFrame(
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
    capped = BalanceMaxima(max_clique=1, max_clique_sex=1)

    suggestion = feasibility.diagnose_balance_caps(
        preferences=preferences,
        students=students,
        groups_to=groups_to,
        not_together=[],
        maxima=capped,
    )

    assert set(suggestion) == {"clique", "clique_sex"}
    assert suggestion["clique"] == {"current": 1, "suggested": 3}
    assert suggestion["clique_sex"] == {"current": 1, "suggested": 2}

    relaxed = engine.solve_within_minimal_relaxation(
        preferences=preferences,
        students=students,
        groups_to=groups_to,
        not_together=[],
        maxima=BalanceMaxima(
            max_clique=suggestion["clique"]["suggested"],
            max_clique_sex=suggestion["clique_sex"]["suggested"],
        ),
    )
    assert relaxed.assignment


def test_balance_cap_diagnosis_does_not_preserve_uncapped_satisfaction_floor():
    """The suggestion restores feasibility without unnecessary cap increases.

    Three students can only go to Rood. Putting the flexible fourth student in
    Blauw gives a valid 3-1 split once the per-year cap becomes 2, and the two
    existing students in Blauw keep the total sizes within 1. Putting that
    fourth student in Rood fulfils their wish, but needs both wider per-year and
    total caps. The diagnosis must suggest the smaller feasible limits, not the
    limits needed to retain the uncapped model's better satisfaction floor.
    """
    groups = ["Rood", "Blauw"]
    group_keys = [matching_key(group) for group in groups]
    students = [
        StudentEntry(
            "Anna",
            "Meisje",
            "A",
            None,
            1,
            excluded_groups=["Blauw"],
        ),
        StudentEntry(
            "Bo",
            "Jongen",
            "B",
            None,
            1,
            excluded_groups=["Blauw"],
        ),
        StudentEntry(
            "Cas",
            "Meisje",
            "C",
            None,
            1,
            excluded_groups=["Blauw"],
        ),
        StudentEntry(
            "Daan",
            "Jongen",
            "D",
            None,
            1,
            preferences=[Preference("Anna", 1.0, PreferenceKind.TOGETHER)],
        ),
    ]
    preference_data = build_preference_data(students, group_keys)
    groups_to = {
        matching_key("Rood"): {"Jongens": 0, "Meisjes": 0},
        matching_key("Blauw"): {"Jongens": 1, "Meisjes": 1},
    }
    maxima = BalanceMaxima(
        max_diff_n_students_year=1,
        max_diff_n_students_total=1,
    )

    suggestion = feasibility.diagnose_balance_caps(
        preferences=preference_data.preferences,
        students=preference_data.students_info,
        groups_to=groups_to,
        not_together=[],
        maxima=maxima,
    )

    assert suggestion == {"diff_year": {"current": 1, "suggested": 2}}
    relaxed = engine.solve_within_minimal_relaxation(
        preferences=preference_data.preferences,
        students=preference_data.students_info,
        groups_to=groups_to,
        not_together=[],
        maxima=BalanceMaxima(
            max_diff_n_students_year=2,
            max_diff_n_students_total=1,
        ),
    )
    assert relaxed.assignment


def test_capped_solver_uses_preference_diagnosis_when_uncapped_is_infeasible():
    """Caps never hide a hard-preference infeasibility from the existing diagnosis."""
    preference_data, target_groups = _infeasible_by_min_satisfaction()
    maxima = BalanceMaxima(max_clique=1)

    assert (
        feasibility.diagnose_balance_caps(
            preferences=preference_data.preferences,
            students=preference_data.students_info,
            groups_to=target_groups.counts,
            not_together=[],
            maxima=maxima,
        )
        is None
    )

    with pytest.raises(errors.FeasibilityError) as exc_info:
        engine.solve_within_minimal_relaxation(
            preferences=preference_data.preferences,
            students=preference_data.students_info,
            groups_to=target_groups.counts,
            not_together=[],
            maxima=maxima,
        )

    assert exc_info.value.code == "infeasible_preferences"
    assert exc_info.value.context == {"case": "min_satisfaction"}
