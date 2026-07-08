"""Tests for satisfaction.py"""

# pylint: disable=protected-access

import pandas as pd
import pytest

from aliexpress.solver import satisfaction


def test_get_satisfaction_integral_basic():
    """Integral from 0 to 1 equals 0.5."""
    assert satisfaction.get_satisfaction_integral(0, 1) == pytest.approx(0.5)


def test_get_satisfaction_integral_decreasing_importance():
    """First preference contributes more than the second."""
    val1 = satisfaction.get_satisfaction_integral(0, 1)
    val2 = satisfaction.get_satisfaction_integral(1, 2)
    assert val1 > val2


def test_get_satisfaction_percentage_basic():
    """Half the weighted preferences honored gives 0.5."""
    assert satisfaction.get_satisfaction_percentage(2.0, 4.0) == pytest.approx(0.5)


def test_get_satisfaction_percentage_zero_max():
    """Returns 1.0 when max_weight is 0 (no preferences to honor)."""
    assert satisfaction.get_satisfaction_percentage(0.0, 0.0) == pytest.approx(1.0)


def test_get_satisfaction_percentage_all_honored():
    """All preferences honored gives 1.0."""
    assert satisfaction.get_satisfaction_percentage(3.0, 3.0) == pytest.approx(1.0)


def test_powerset_and_unique_sums_hidden_helpers():
    """_all_unique_sums returns all subset sums."""
    iterable = [1, 1, 0.5, 4]
    sums = satisfaction._all_unique_sums(iterable)
    assert sums == {0, 0.5, 1, 1.5, 2, 2.5, 4, 4.5, 5, 5.5, 6, 6.5}


def test_achievable_weighted_levels_simple():
    """Correct achievable levels for a single student with two preferences."""
    df = pd.DataFrame(
        {
            "Waarde": ["A", "B"],
            "Gewicht": [1, 2],
        },
        index=pd.MultiIndex.from_product(
            [["s1", "s1"], ["Graag met"]], names=["Leerling", "TypeWens"]
        ),
    )
    result = satisfaction._achievable_weighted_levels(df)
    assert result == {0, 1, 2, 3}


def test_calculate_added_satisfaction_monotonic():
    """Satisfaction scores are positive and match the integral values."""
    df = pd.DataFrame(
        {
            "Waarde": ["A", "B"],
            "Gewicht": [1, 2],
        },
        index=pd.MultiIndex.from_product(
            [["s1", "s1"], ["Graag met"]], names=["Leerling", "TypeWens"]
        ),
    )
    added = satisfaction.calculate_added_satisfaction(df)
    assert all(isinstance(v, float) for v in added.values())
    assert all(v > 0 for v in added.values())

    expected = {1: 0.5, 2: 0.25, 3: 0.125}
    assert added == expected


def test_normalize_and_bound_avoid_wish_violated_lower_bound():
    """A single violated avoid-wish scores -1.0 (the jump lower bound)."""
    assert satisfaction._normalize_and_bound(-1, 0, -1) == pytest.approx(-1.0)


def test_normalize_and_bound_avoid_wish_honored_upper_bound():
    """An avoid-wish that is honored (not violated) scores +1.0 (the jump upper bound)."""
    assert satisfaction._normalize_and_bound(0, 0, -1) == pytest.approx(1.0)


def test_normalize_and_bound_avoid_wishes_saturating_band():
    """Multiple avoid-wishes saturate towards -1.0 as more are violated."""
    assert satisfaction._normalize_and_bound(-1, 0, -3) == pytest.approx(-1 / 7)
    assert satisfaction._normalize_and_bound(-3, 0, -3) == pytest.approx(-1.0)


def test_normalize_and_bound_positive_and_mixed_unchanged():
    """Students with positive wishes (incl. mixed) normalize by F(best), unchanged."""
    assert satisfaction._normalize_and_bound(1, 1, 0) == pytest.approx(1.0)
    assert satisfaction._normalize_and_bound(0, 2, 0) == pytest.approx(0.0)
    assert satisfaction._normalize_and_bound(1, 2, 0) == pytest.approx(2 / 3)
    assert satisfaction._normalize_and_bound(-1, 2, -3) == pytest.approx(-1 / 0.75)
    assert satisfaction._normalize_and_bound(0, 0, 0) == pytest.approx(1.0)
