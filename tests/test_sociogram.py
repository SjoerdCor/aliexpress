"""Tests for sociogram.SociogramMaker."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import math

import pytest
from pandas.testing import assert_frame_equal

from aliexpress import sociogram
from aliexpress.data import datareader

GROUPS = ["blauw", "groen", "geel", "oranje"]
PREFS_PATH = "testdata/voorkeuren_klein.xlsx"


@pytest.fixture()
def preference_data():
    """PreferenceData built from the small test Excel."""
    processor = datareader.VoorkeurenProcessor(PREFS_PATH)
    processor.process(all_to_groups=GROUPS)
    return processor.to_preference_data()


def _students_info_equal(a: dict, b: dict) -> bool:
    """NaN-aware equality for students_info dicts."""
    if a.keys() != b.keys():
        return False
    for key in a:
        for field in a[key]:
            av, bv = a[key][field], b[key][field]
            nan_a = isinstance(av, float) and math.isnan(av)
            nan_b = isinstance(bv, float) and math.isnan(bv)
            if nan_a != nan_b:
                return False
            if not nan_a and av != bv:
                return False
    return True


def test_from_preference_data_matches_constructor(preference_data):
    """from_preference_data gives same preferences and students_info as the file constructor."""
    via_file = sociogram.SociogramMaker(PREFS_PATH, GROUPS)
    via_data = sociogram.SociogramMaker.from_preference_data(preference_data)

    assert_frame_equal(via_data.preferences, via_file.preferences)
    assert _students_info_equal(via_data.students_info, via_file.students_info)


def test_matplotlib_backend_is_agg():
    """sociogram.py must set the Agg backend to avoid Tk/display errors on headless servers."""
    import matplotlib  # pylint: disable=import-outside-toplevel

    assert matplotlib.get_backend().lower() == "agg"
