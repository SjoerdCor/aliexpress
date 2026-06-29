"""Tests for pulp_thresholds.py — generic big-M indicator constraints."""

import pulp

from aliexpress.solver import pulp_thresholds


def test_apply_threshold_constraint_activates_at_threshold():
    """Single threshold: var becomes 1 when value reaches the threshold."""
    prob = pulp.LpProblem("test_single", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    thr_var = pulp.LpVariable("thr", cat="Binary")

    pulp_thresholds.apply_threshold_constraint(prob, x, 3.0, thr_var)

    prob += x == 3.5
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert thr_var.value() == 1


def test_apply_threshold_constraint_inactive_below_threshold():
    """Single threshold: var stays 0 when value is below the threshold."""
    prob = pulp.LpProblem("test_single_below", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    thr_var = pulp.LpVariable("thr", cat="Binary")

    pulp_thresholds.apply_threshold_constraint(prob, x, 3.0, thr_var)

    prob += x == 1.0
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert thr_var.value() == 0


def test_apply_threshold_constraint_creates_variable_when_none():
    """apply_threshold_constraint creates and returns a binary variable when not given one."""
    prob = pulp.LpProblem("test_auto_var", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)

    thr_var = pulp_thresholds.apply_threshold_constraint(prob, x, 2.0)
    assert thr_var is not None
    assert thr_var.lowBound == 0
    assert thr_var.upBound == 1


def test_apply_threshold_constraint_invalid_sense():
    """apply_threshold_constraint raises ValueError for unknown sense."""
    prob = pulp.LpProblem("test_sense", pulp.LpMinimize)
    x = pulp.LpVariable("x")
    try:
        pulp_thresholds.apply_threshold_constraint(prob, x, 1.0, sense="!=")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_apply_threshold_constraints_positive():
    """Test threshold constraints are added correctly for positive threshold"""
    prob = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    thresholds = [1, 2]
    threshold_vars = {t: pulp.LpVariable(f"thr_{t}", cat="Binary") for t in thresholds}

    pulp_thresholds.apply_threshold_constraints(prob, x, thresholds, threshold_vars)

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

    pulp_thresholds.apply_threshold_constraints(prob, x, thresholds, threshold_vars)
    prob += x == -1.5
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert threshold_vars[-1].value() == 1
    assert threshold_vars[-2].value() == 0
