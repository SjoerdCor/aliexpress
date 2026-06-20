"""Tests for building PreferenceData from web-form wishes (Stap 2).

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

from aliexpress.errors import ValidationError
from aliexpress.preferences_data import PreferenceData
from aliexpress.preferences_form import StudentWishes, Wish, build_preference_data


def _student(**overrides) -> StudentWishes:
    """Construct a StudentWishes with sensible defaults; override any field by keyword."""
    base = StudentWishes(
        leerling="John",
        geslacht="Jongen",
        stamgroep="Rood",
        minimale_tevredenheid=None,
        graag_met=[],
        liever_niet_met=[],
        niet_in=[],
    )
    return replace(base, **overrides)


def test_valid_wishes_build_preference_data():
    """A graag-met + liever-niet-met wish yields the expected long rows and negation.

    'Liever niet met' must carry a negative weight (post-negation), and the wide
    input_sheet must hold the original pre-negation wishes keyed by matching_key.
    """
    students = [
        _student(
            leerling="John",
            graag_met=[Wish(naam="Jane", gewicht=2.0)],
            liever_niet_met=[Wish(naam="Blauw", gewicht=3.0)],
        ),
        _student(leerling="Jane", geslacht="Meisje", stamgroep="Blauw"),
    ]
    data = build_preference_data(students, all_to_groups=["rood", "blauw"])

    # Long-format preferences are post-negation: "Liever niet met" collapses into
    # "Graag met" with a negative weight (toggle_negative_weights), appended after the
    # positive graag-met rows and renumbered.
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

    # Wide input_sheet holds the pre-negation wish keyed by matching_key.
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


def test_niet_in_up_to_groups_minus_one_is_allowed():
    """A student may avoid all but one group; avoiding every group is rejected."""
    groups = ["rood", "blauw", "groen"]

    ok = build_preference_data(
        [_student(niet_in=["Rood", "Blauw"])], all_to_groups=groups
    )
    assert ok.preferences.xs("Niet in", level="TypeWens").shape[0] == 2

    with pytest.raises(ValidationError) as exc:
        build_preference_data(
            [_student(niet_in=["Rood", "Blauw", "Groen"])], all_to_groups=groups
        )
    assert exc.value.code == "too_many_niet_in_form"


def test_duplicate_target_within_one_student_is_rejected():
    """The same target twice for one student fails the shared uniqueness check."""
    students = [
        _student(
            leerling="John",
            graag_met=[Wish("Jane", 1.0), Wish("Jane", 2.0)],
        ),
        _student(leerling="Jane", geslacht="Meisje", stamgroep="Blauw"),
    ]
    with pytest.raises(pa.errors.SchemaError) as exc:
        build_preference_data(students, all_to_groups=["rood", "blauw"])
    assert exc.value.check.name == "duplicated_values_preferences"


def test_unknown_target_is_rejected():
    """A target that is neither a known student nor a known group is rejected."""
    # 'graag met' a non-existent name.
    with pytest.raises(pa.errors.SchemaError) as exc:
        build_preference_data(
            [_student(graag_met=[Wish("Ghost", 1.0)])],
            all_to_groups=["rood", "blauw"],
        )
    assert exc.value.check.name == "invalid_values_preferences"

    # 'niet in' a name that is a student, not a group.
    with pytest.raises(pa.errors.SchemaError) as exc:
        build_preference_data(
            [
                _student(leerling="John", niet_in=["Jane"]),
                _student(leerling="Jane", geslacht="Meisje", stamgroep="Blauw"),
            ],
            all_to_groups=["rood", "blauw"],
        )
    assert exc.value.check.name == "invalid_values_preferences"


def test_non_positive_weight_is_rejected():
    """A weight of zero or below is rejected with a friendly form error."""
    students = [
        _student(leerling="John", graag_met=[Wish("Jane", 0.0)]),
        _student(leerling="Jane", geslacht="Meisje", stamgroep="Blauw"),
    ]
    with pytest.raises(ValidationError) as exc:
        build_preference_data(students, all_to_groups=["rood", "blauw"])
    assert exc.value.code == "invalid_gewicht_form"


def test_minimale_tevredenheid_above_one_is_rejected():
    """MinimaleTevredenheid must be <= 1; a higher value is rejected."""
    with pytest.raises(ValidationError) as exc:
        build_preference_data(
            [_student(minimale_tevredenheid=1.5)], all_to_groups=["rood", "blauw"]
        )
    assert exc.value.code == "invalid_min_tevredenheid_form"


def test_negative_minimale_tevredenheid_is_allowed():
    """A negative MinimaleTevredenheid is allowed (liever-niet-met can go negative)."""
    data = build_preference_data(
        [_student(minimale_tevredenheid=-2.0)], all_to_groups=["rood", "blauw"]
    )
    assert data.students_info["john"]["MinimaleTevredenheid"] == -2.0


def test_student_without_wishes_is_allowed():
    """A selected student who entered no wishes still produces valid PreferenceData."""
    data = build_preference_data(
        [_student(leerling="John")], all_to_groups=["rood", "blauw"]
    )
    assert data.preferences.empty
    assert data.students_info["john"]["Jongen/meisje"] == "Jongen"
    assert data.student_display["john"] == "John"


def test_unlimited_graag_met_is_accepted_and_round_trips():
    """Eight 'graag met' wishes are accepted (no fixed column cap) and survive JSON."""
    targets = [f"Mate{i}" for i in range(8)]
    students = [_student(leerling="John", graag_met=[Wish(t, 1.0) for t in targets])]
    students += [
        _student(leerling=t, geslacht="Meisje", stamgroep="Blauw") for t in targets
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
