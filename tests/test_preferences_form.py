"""Tests for building PreferenceData from web-form preferences (Stap 2).

The builder turns small, readable dataclasses (one per student) into the canonical
``PreferenceData`` contract the solver consumes, reusing the shared long-format
validation from ``datareader``. It is storage-agnostic: it returns an object, it does
not read or write files. Only synthetic data is used here, never real student data.
"""

import math
from dataclasses import replace

import pandas as pd
import pandera.pandas as pa
import pytest

from aliexpress.data.preferences_data import PreferenceData
from aliexpress.data.preferences_form import (
    Preference,
    PreferenceKind,
    StudentEntry,
    build_preference_data,
)
from aliexpress.errors import ValidationError


def _student(**overrides) -> StudentEntry:
    """Construct a StudentEntry with sensible defaults; override any field by keyword."""
    base = StudentEntry(
        student="John",
        sex="Jongen",
        origin_group="Rood",
        min_satisfaction=None,
        preferences=[],
        excluded_groups=[],
    )
    return replace(base, **overrides)


def _together(target, weight):
    """A 'graag met' preference."""
    return Preference(target=target, weight=weight, kind=PreferenceKind.TOGETHER)


def _apart(target, weight):
    """A 'liever niet met' preference."""
    return Preference(target=target, weight=weight, kind=PreferenceKind.APART)


def test_valid_preferences_build_preference_data():
    """A together + apart preference yields the expected long rows and negation.

    The apart preference must carry a negative weight (post-negation), and the wide
    input_sheet must hold the original pre-negation preferences keyed by matching_key.
    """
    students = [
        _student(
            student="John",
            preferences=[_together("Jane", 2.0), _apart("Blauw", 3.0)],
        ),
        _student(student="Jane", sex="Meisje", origin_group="Blauw"),
    ]
    data = build_preference_data(students, all_to_groups=["rood", "blauw"])

    # Long-format preferences are post-negation: "Liever niet met" collapses into
    # "Graag met" with a negative weight (toggle_negative_weights), appended after the
    # positive together rows and renumbered.
    prefs = data.preferences
    assert prefs.loc[("john", "Graag met", 1.0), "Waarde"] == "jane"
    assert prefs.loc[("john", "Graag met", 1.0), "Gewicht"] == 2.0
    assert prefs.loc[("john", "Graag met", 2.0), "Waarde"] == "blauw"
    assert prefs.loc[("john", "Graag met", 2.0), "Gewicht"] == -3.0
    assert "Liever niet met" not in prefs.index.get_level_values("TypeWens")

    # Display maps keep the names as entered.
    assert data.student_display["john"] == "John"
    assert data.student_display["jane"] == "Jane"
    assert data.stamgroep_display["rood"] == "Rood"

    # students_info mirrors get_students_meta_info's shape.
    assert data.students_info["john"]["Jongen/meisje"] == "Jongen"
    assert data.students_info["john"]["Stamgroep"] == "rood"
    assert math.isnan(data.students_info["john"]["MinimaleTevredenheid"])

    # Wide input_sheet holds the pre-negation preference keyed by matching_key.
    sheet = data.input_sheet
    assert sheet.loc["john", ("Graag met", 1.0, "Waarde")] == "jane"
    assert sheet.loc["john", ("Graag met", 1.0, "Gewicht")] == 2.0
    assert sheet.loc["john", ("Liever niet met", 1.0, "Waarde")] == "blauw"
    assert sheet.loc["john", ("Liever niet met", 1.0, "Gewicht")] == 3.0
    # The info columns are keyed by NaN sub-levels (mirrors VoorkeurenProcessor.input).
    info_col = next(c for c in sheet.columns if c[0] == "Jongen/meisje")
    assert sheet.loc["john", info_col] == "Jongen"
    assert isinstance(sheet.columns, pd.MultiIndex)
    assert sheet.columns.names == ["TypeWens", "Nr", "TypeWaarde"]


def test_excluded_groups_up_to_groups_minus_one_is_allowed():
    """A student may avoid all but one group; avoiding every group is rejected."""
    groups = ["rood", "blauw", "groen"]

    ok = build_preference_data(
        [_student(excluded_groups=["Rood", "Blauw"])], all_to_groups=groups
    )
    assert ok.preferences.xs("Niet in", level="TypeWens").shape[0] == 2

    with pytest.raises(ValidationError) as exc:
        build_preference_data(
            [_student(excluded_groups=["Rood", "Blauw", "Groen"])], all_to_groups=groups
        )
    assert exc.value.code == "too_many_niet_in_form"


def test_duplicate_target_within_one_student_is_rejected():
    """The same target twice for one student fails the shared uniqueness check."""
    students = [
        _student(
            student="John",
            preferences=[_together("Jane", 1.0), _together("Jane", 2.0)],
        ),
        _student(student="Jane", sex="Meisje", origin_group="Blauw"),
    ]
    with pytest.raises(pa.errors.SchemaError) as exc:
        build_preference_data(students, all_to_groups=["rood", "blauw"])
    assert exc.value.check.name == "duplicated_values_preferences"


def test_duplicate_group_target_is_allowed():
    """The same group twice for one student is allowed: group preferences stack (ADR 0004)."""
    students = [
        _student(
            student="John",
            preferences=[_together("Blauw", 2.0), _together("Blauw", 0.5)],
        ),
    ]
    data = build_preference_data(students, all_to_groups=["rood", "blauw"])
    rows = data.preferences.xs(("john", "Graag met"), level=["Leerling", "TypeWens"])
    assert rows.shape[0] == 2
    assert sorted(rows["Gewicht"].tolist()) == [0.5, 2.0]


def test_group_target_in_both_directions_is_allowed():
    """A group may be both 'graag met' and 'liever niet met' — they stack/counteract."""
    students = [
        _student(
            student="John",
            preferences=[_together("Blauw", 2.0), _apart("Blauw", 1.0)],
        ),
    ]
    data = build_preference_data(students, all_to_groups=["rood", "blauw"])
    # Post-negation both collapse into "Graag met": +2 and -1.
    weights = sorted(data.preferences.xs("john", level="Leerling")["Gewicht"].tolist())
    assert weights == [-1.0, 2.0]


def test_same_student_in_both_lists_is_rejected():
    """The same student as both 'graag met' and 'liever niet met' is contradictory."""
    students = [
        _student(
            student="John",
            preferences=[_together("Jane", 1.0), _apart("Jane", 1.0)],
        ),
        _student(student="Jane", sex="Meisje", origin_group="Blauw"),
    ]
    with pytest.raises(pa.errors.SchemaError) as exc:
        build_preference_data(students, all_to_groups=["rood", "blauw"])
    assert exc.value.check.name == "duplicated_values_preferences"


def test_unknown_target_is_rejected():
    """A target that is neither a known student nor a known group is rejected."""
    # 'graag met' a non-existent name.
    with pytest.raises(pa.errors.SchemaError) as exc:
        build_preference_data(
            [_student(preferences=[_together("Ghost", 1.0)])],
            all_to_groups=["rood", "blauw"],
        )
    assert exc.value.check.name == "invalid_values_preferences"

    # 'niet in' a name that is a student, not a group.
    with pytest.raises(pa.errors.SchemaError) as exc:
        build_preference_data(
            [
                _student(student="John", excluded_groups=["Jane"]),
                _student(student="Jane", sex="Meisje", origin_group="Blauw"),
            ],
            all_to_groups=["rood", "blauw"],
        )
    assert exc.value.check.name == "invalid_values_preferences"


def test_non_positive_weight_is_rejected_at_construction():
    """A Preference enforces weight > 0 as an invariant (the route flashes a message)."""
    with pytest.raises(ValueError):
        Preference(target="Jane", weight=0.0, kind=PreferenceKind.TOGETHER)


def test_min_satisfaction_above_one_is_rejected():
    """min_satisfaction must be <= 1 (100%); a higher value is rejected."""
    with pytest.raises(ValidationError) as exc:
        build_preference_data(
            [_student(min_satisfaction=1.5)], all_to_groups=["rood", "blauw"]
        )
    assert exc.value.code == "invalid_min_tevredenheid_form"


def test_negative_min_satisfaction_is_allowed():
    """A negative min_satisfaction is allowed (an apart preference can go negative)."""
    data = build_preference_data(
        [_student(min_satisfaction=-2.0)], all_to_groups=["rood", "blauw"]
    )
    assert data.students_info["john"]["MinimaleTevredenheid"] == -2.0


def test_student_without_preferences_is_allowed():
    """A selected student who entered no preferences still produces valid PreferenceData."""
    data = build_preference_data(
        [_student(student="John")], all_to_groups=["rood", "blauw"]
    )
    assert data.preferences.empty
    assert data.students_info["john"]["Jongen/meisje"] == "Jongen"
    assert data.student_display["john"] == "John"


def test_unlimited_together_is_accepted_and_round_trips():
    """Eight together preferences are accepted (no fixed column cap) and survive JSON."""
    targets = [f"Mate{i}" for i in range(8)]
    students = [
        _student(student="John", preferences=[_together(t, 1.0) for t in targets])
    ]
    students += [
        _student(student=t, sex="Meisje", origin_group="Blauw") for t in targets
    ]

    data = build_preference_data(students, all_to_groups=["rood", "blauw"])
    assert (
        data.preferences.xs(
            ("john", "Graag met"), level=["Leerling", "TypeWens"]
        ).shape[0]
        == 8
    )

    restored = PreferenceData.from_json(data.to_json())
    pd.testing.assert_frame_equal(restored.preferences, data.preferences)
    pd.testing.assert_frame_equal(restored.input_sheet, data.input_sheet)
