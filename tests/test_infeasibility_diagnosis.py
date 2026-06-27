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


def _infeasible_by_extra_zekerheid():
    """Smallest synthetic infeasible input on the automatic path.

    Anna must be with Bo (Extra zekerheid 'belangrijkste voorkeur'), but their
    Niet-in exclusions force them into disjoint groups, so the floor can never be
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
    preference_data, target_groups = _infeasible_by_extra_zekerheid()

    with pytest.raises(errors.FeasibilityError) as exc_info:
        distribute_students_from_data(preference_data, target_groups, groupbalance=None)

    exc = exc_info.value
    assert exc.code == "infeasible_preferences"
    message = to_validation_message(exc)
    assert "extra zekerheid" in message.lower()
    assert "niet-samen" in message.lower()
