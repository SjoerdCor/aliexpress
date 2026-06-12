"""Unit tests for solutions.py"""

import pandas as pd

from aliexpress.solutions import SolutionAnalyzer


def test_probvars_to_series_empty_returns_empty_multiindex():
    """Empty prob_vars must not crash on .str accessor (pandas >= 2 issue)."""
    # pylint: disable-next=protected-access  # testing an internal static method directly
    result = SolutionAnalyzer._probvars_to_series({}, "Satisfied", "per_group")
    assert isinstance(result, pd.Series)
    assert len(result) == 0
    assert result.index.names == ["student", "Nr"]


def test_probvars_to_series_extracts_student_and_nr():
    """Non-empty case: student name and preference number are extracted correctly."""
    prob_vars = {
        "Satisfied_('Anna',_1.0)": _FakeVar(1.0),
        "Satisfied_('Bram',_2.0)": _FakeVar(0.0),
        "Satisfied_per_group_ignored": _FakeVar(1.0),
    }
    # pylint: disable-next=protected-access  # testing an internal static method directly
    result = SolutionAnalyzer._probvars_to_series(prob_vars, "Satisfied", "per_group")
    assert set(result.index.get_level_values("student")) == {"Anna", "Bram"}
    assert len(result) == 2


# pylint: disable=too-few-public-methods  # minimal test double: only needs value()
class _FakeVar:
    def __init__(self, val):
        self._val = val

    def value(self):
        """Return the stored value."""
        return self._val
