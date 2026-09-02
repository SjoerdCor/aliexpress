"""Unit tests for main.py orchestration helpers."""

import logging

import pytest

from aliexpress import errors
from aliexpress.data.datareader import GroupCounts, matching_key
from aliexpress.data.preferences_form import (
    Preference,
    PreferenceKind,
    StudentEntry,
    build_preference_data,
)
from aliexpress.main import (
    _log_initial_state,
    build_input_summary,
    distribute_students_from_data,
)


def test_log_initial_state_reports_stamgroep_as_entered(monkeypatch):
    """The initial-state log lines show Stamgroep names as entered, not matching keys."""
    groups_to = {"blauw": {"Jongens": 1, "Meisjes": 1}}
    students_info = {
        "jan": {"Jongen/meisje": "Jongen", "Stamgroep": "klasa"},
        "eva": {"Jongen/meisje": "Meisje", "Stamgroep": "klasa"},
        "tom": {"Jongen/meisje": "Jongen", "Stamgroep": "annes"},
    }
    stamgroep_display = {"klasa": "Klas A", "annes": "Anne's groep"}

    logged: list[str] = []
    monkeypatch.setattr(
        logging.getLogger("aliexpress.main"),
        "info",
        lambda msg, *args: logged.append(msg % args),
    )

    _log_initial_state(groups_to, students_info, stamgroep_display)

    full_log = "\n".join(logged)
    assert "Klas A" in full_log
    assert "Anne's groep" in full_log
    # The internal matching keys must not leak into the log messages.
    assert "klasa" not in full_log


def test_build_input_summary_counts_display_groups_not_matching_keys():
    """The input overview counts display Stamgroepen, not the internal matching keys."""
    groups_to = {
        "blauw": {"Jongens": 1, "Meisjes": 1},
        "rood": {"Jongens": 1, "Meisjes": 1},
    }
    students_info = {
        "jan": {"Jongen/meisje": "Jongen", "Stamgroep": "klasa", "Jaarlaag": 6},
        "eva": {"Jongen/meisje": "Meisje", "Stamgroep": "klasa2", "Jaarlaag": 6},
        "tom": {"Jongen/meisje": "Jongen", "Stamgroep": "annes", "Jaarlaag": 7},
    }
    # jan and eva sit in two *different* matching keys ("klasa", "klasa2") that map to
    # the *same* display name "Klas A". Counting matching keys would give three source
    # groups; grouping by display name collapses them to "Klas A" with 2 students.
    stamgroep_display = {"klasa": "Klas A", "klasa2": "Klas A", "annes": "Anne's groep"}

    summary = build_input_summary(groups_to, students_info, stamgroep_display)

    assert summary.n_students == 3
    assert summary.n_boys == 2
    assert summary.n_girls == 1
    assert summary.source_groups == {"Klas A": 2, "Anne's groep": 1}
    assert summary.n_target_groups == 2
    assert summary.years == [6, 7]


def test_distribution_translates_detailed_conflict_keys_at_main_boundary(monkeypatch):
    """Solver keys become entered names while the technical error stays unchanged."""
    students = [
        StudentEntry(
            "Piet Jansen",
            "Jongen",
            "Klas A",
            1.0,
            preferences=[
                Preference("Sam de Vries", 1.0, PreferenceKind.TOGETHER),
                Preference("Blauw", 2.0, PreferenceKind.APART),
            ],
            excluded_groups=["Rood"],
        ),
        StudentEntry("Sam de Vries", "Meisje", "Klas A", None),
    ]
    blue, red = matching_key("Blauw"), matching_key("Rood")
    preference_data = build_preference_data(
        students,
        [blue, red],
        unique_name={
            matching_key("Piet Jansen"): "Piet",
            matching_key("Sam de Vries"): "Sam",
        },
    )
    target_groups = GroupCounts(
        counts={blue: {"Jongens": 0, "Meisjes": 0}, red: {"Jongens": 0, "Meisjes": 0}},
        display={blue: "Blauw", red: "Rood"},
    )
    technical_message = "Hard preference constraints are mutually infeasible"
    detail = {
        "case": "detailed",
        "conflict": {
            "conditions": [
                {
                    "type": "minimum_satisfaction",
                    "student": matching_key("Piet Jansen"),
                    "floor": 1.0,
                    "preferences": [
                        {
                            "kind": "Graag met",
                            "target": matching_key("Sam de Vries"),
                            "weight": 1.0,
                        },
                        {
                            "kind": "Liever niet met",
                            "target": blue,
                            "weight": -2.0,
                        },
                    ],
                },
                {
                    "type": "forbidden_group",
                    "student": matching_key("Piet Jansen"),
                    "group": red,
                },
            ]
        },
    }

    def fail(**_kwargs):
        raise errors.FeasibilityError(
            "infeasible_preferences", detail, technical_message=technical_message
        )

    monkeypatch.setattr("aliexpress.main.engine.solve_within_minimal_relaxation", fail)

    with pytest.raises(errors.FeasibilityError) as exc_info:
        distribute_students_from_data(preference_data, target_groups)

    translated = exc_info.value.context["conflict"]["conditions"]
    assert translated[0]["student"] == "Piet"
    assert translated[0]["preferences"][0]["target"] == "Sam"
    assert translated[0]["preferences"][1]["target"] == "Blauw"
    assert translated[1] == {
        "type": "forbidden_group",
        "student": "Piet",
        "group": "Rood",
    }
    assert exc_info.value.technical_message == technical_message
