"""Tests for the friendly diagnosis raised when preferences are infeasible.

On the automatic path (``groupbalance=None``) the relaxation-budget LP can be
Infeasible because the hard preference constraints (Extra zekerheid and/or
Niet-samen) contradict each other.  Instead of a raw ``ValueError`` the solver
must raise a ``FeasibilityError`` that renders as a friendly Dutch message.
"""

import pytest

from aliexpress import errors
from aliexpress.datareader import GroupCounts, matching_key
from aliexpress.main import distribute_students_from_data
from aliexpress.preferences_form import (
    Preference,
    PreferenceKind,
    StudentEntry,
    build_preference_data,
)
from aliexpress.validation_messages import to_validation_message


def _infeasible_by_min_satisfaction():
    """Smallest synthetic infeasible input on the automatic path.

    Anna must be with Bo (minimal satisfaction at 'belangrijkste voorkeur'), but their
    forbidden-group exclusions force them into disjoint groups, so the floor can never be
    met.  Four filler students keep the balance constraints from dominating.
    """
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


def test_infeasible_auto_path_raises_friendly_error():
    """Infeasible preferences raise a FeasibilityError that renders in Dutch."""
    preference_data, target_groups = _infeasible_by_min_satisfaction()

    with pytest.raises(errors.FeasibilityError) as exc_info:
        distribute_students_from_data(preference_data, target_groups, groupbalance=None)

    assert exc_info.value.code == "infeasible_preferences"


def test_min_satisfaction_family_is_reported():
    """Infeasibility caused by minimal satisfaction alone is diagnosed as that family."""
    preference_data, target_groups = _infeasible_by_min_satisfaction()

    with pytest.raises(errors.FeasibilityError) as exc_info:
        distribute_students_from_data(preference_data, target_groups, groupbalance=None)

    assert exc_info.value.context["case"] == "min_satisfaction"
    assert "extra zekerheid" in to_validation_message(exc_info.value).lower()


def _infeasible_by_not_together():
    """Smallest input infeasible because of a not-together rule alone.

    Ali, Bram and Cis can each only go to Rood (forbidden-group exclusions block the
    other two groups), but a not-together rule allows at most one of them per group.  No
    minimal satisfaction is involved.
    """
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
    not_together = [{"group": {"Ali", "Bram", "Cis"}, "Max_aantal_samen": 1}]
    return preference_data, target_groups, not_together


def test_not_together_family_is_reported():
    """Infeasibility caused by a not-together rule alone is diagnosed as that family."""
    preference_data, target_groups, not_together = _infeasible_by_not_together()

    with pytest.raises(errors.FeasibilityError) as exc_info:
        distribute_students_from_data(
            preference_data, target_groups, not_together=not_together, groupbalance=None
        )

    assert exc_info.value.context["case"] == "not_together"
    assert "niet-samen" in to_validation_message(exc_info.value).lower()
