"""Tests for the PreferenceData datacontract and its lossless JSON round-trip."""

import math

import pandas as pd
from pandas.testing import assert_frame_equal

from aliexpress.preferences_data import PreferenceData


def _sample_preferences() -> pd.DataFrame:
    """A small long-format preferences frame with a positive and a negative weight.

    Index: (Leerling, TypeWens, Nr); columns: Waarde, Gewicht. Mirrors the structure
    produced by ``VoorkeurenProcessor`` (Nr is a float level after stacking).
    """
    index = pd.MultiIndex.from_tuples(
        [
            ("aap", "Graag met", 1.0),
            ("aap", "Liever niet met", 1.0),
        ],
        names=["Leerling", "TypeWens", "Nr"],
    )
    return pd.DataFrame(
        {"Waarde": ["noot", "mies"], "Gewicht": [1.0, -2.0]},
        index=index,
    )


def _sample_students_info() -> dict:
    """Meta info covering the full diversity get_students_meta_info can yield.

    ``aap`` has an explicit minimum satisfaction; ``noot`` left it blank (NaN, the
    common case); both sexes and distinct stamgroepen are represented.
    """
    return {
        "aap": {
            "MinimaleTevredenheid": 0.5,
            "Jongen/meisje": "Jongen",
            "Stamgroep": "rood",
        },
        "noot": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Meisje",
            "Stamgroep": "blauw",
        },
    }


def _assert_students_info_equal(actual: dict, expected: dict) -> None:
    """Compare meta dicts with NaN-aware equality (NaN == NaN must hold here)."""
    assert actual.keys() == expected.keys()
    for key in expected:
        act, exp = actual[key], expected[key]
        assert act.keys() == exp.keys()
        for field, exp_value in exp.items():
            act_value = act[field]
            if isinstance(exp_value, float) and math.isnan(exp_value):
                assert isinstance(act_value, float) and math.isnan(act_value)
            else:
                assert act_value == exp_value


def test_preference_data_round_trips_through_json():
    """A PreferenceData survives a to_json/from_json round-trip unchanged."""
    original = PreferenceData(
        preferences=_sample_preferences(),
        students_info=_sample_students_info(),
        student_display={"aap": "Aap", "noot": "Noot", "mies": "Mies"},
        stamgroep_display={"rood": "Rood", "blauw": "Blauw"},
    )

    restored = PreferenceData.from_json(original.to_json())

    assert_frame_equal(restored.preferences, original.preferences)
    _assert_students_info_equal(restored.students_info, original.students_info)
    assert restored.student_display == original.student_display
    assert restored.stamgroep_display == original.stamgroep_display
