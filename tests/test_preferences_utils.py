"""Test preferences_utils.py"""

# pylint: disable=protected-access

import pandas as pd
import pulp
import pytest

from aliexpress import preferences_utils


def test_get_satisfaction_integral_basic():
    """Test satisfaction integral"""
    assert preferences_utils.get_satisfaction_integral(0, 1) == pytest.approx(0.5)


def test_get_satisfaction_integral_decreasing_importance():
    """Test satisfaction gives more weight to first preferences"""
    val1 = preferences_utils.get_satisfaction_integral(0, 1)
    val2 = preferences_utils.get_satisfaction_integral(1, 2)
    assert val1 > val2


def test_powerset_and_unique_sums_hidden_helpers():
    """Test finding all unique sums"""
    iterable = [1, 1, 0.5, 4]
    sums = preferences_utils._all_unique_sums(iterable)
    assert sums == {0, 0.5, 1, 1.5, 2, 2.5, 4, 4.5, 5, 5.5, 6, 6.5}


def test_get_possible_weighted_preferences_simple():
    """Test correct possible weighted preferences for a single student"""
    df = pd.DataFrame(
        {
            "Waarde": ["A", "B"],
            "Gewicht": [1, 2],
        },
        index=pd.MultiIndex.from_product(
            [["s1", "s1"], ["Graag met"]], names=["Leerling", "TypeWens"]
        ),
    )
    result = preferences_utils.get_possible_weighted_preferences(df)
    assert result == {0, 1, 2, 3}


def test_calculate_added_satisfaction_monotonic():
    """Test satisfaction is calculated correctly"""
    df = pd.DataFrame(
        {
            "Waarde": ["A", "B"],
            "Gewicht": [1, 2],
        },
        index=pd.MultiIndex.from_product(
            [["s1", "s1"], ["Graag met"]], names=["Leerling", "TypeWens"]
        ),
    )
    added = preferences_utils.calculate_added_satisfaction(df)
    print(added)
    assert all(isinstance(v, float) for v in added.values())
    assert all(v > 0 for v in added.values())

    expected = {1: 0.5, 2: 0.25, 3: 0.125}
    assert added == expected


def test_apply_threshold_constraints_positive():
    """Test threshold constraints are added correctly for positive threshold"""
    prob = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    thresholds = [1, 2]
    threshold_vars = {t: pulp.LpVariable(f"thr_{t}", cat="Binary") for t in thresholds}

    preferences_utils.apply_threshold_constraints(prob, x, thresholds, threshold_vars)

    prob += x == 1.5
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert threshold_vars[1].value() == 1
    assert threshold_vars[2].value() == 0


def test_apply_threshold_constraints_negative():
    """Test threshold works negative thresholds"""
    prob = pulp.LpProblem("test_neg", pulp.LpMinimize)
    x = pulp.LpVariable("x")
    thresholds = [-1, -2]
    threshold_vars = {t: pulp.LpVariable(f"thr_{t}", cat="Binary") for t in thresholds}

    preferences_utils.apply_threshold_constraints(prob, x, thresholds, threshold_vars)
    prob += x == -1.5
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert threshold_vars[-1].value() == 1
    assert threshold_vars[-2].value() == 0
