"""Tests for _balance.py: solver construction and warm starts."""

# pylint: disable=protected-access

from unittest.mock import patch

import highspy
import pulp
import pytest

from aliexpress.solver import _balance


def _small_problem() -> tuple[pulp.LpProblem, dict]:
    prob = pulp.LpProblem("warm", pulp.LpMaximize)
    x = pulp.LpVariable("x", cat="Binary")
    y = pulp.LpVariable("y", cat="Binary")
    prob += x + y <= 1
    prob += x + 2 * y  # objective
    return prob, {"x": x, "y": y}


def test_initial_solution_none_when_values_missing():
    """Without previous values there is nothing to warm-start from."""
    prob, _ = _small_problem()
    assert _balance._initial_solution(prob) is None


def test_initial_solution_follows_variable_order():
    """A complete value set is returned in lp.variables() (column) order."""
    prob, variables = _small_problem()
    variables["x"].setInitialValue(0)
    variables["y"].setInitialValue(1)
    expected = [v.varValue for v in prob.variables()]
    assert _balance._initial_solution(prob) == expected


def test_warm_start_solver_passes_solution_to_highs():
    """WarmStartHiGHS hands the current values to HiGHS as a MIP start."""
    prob, variables = _small_problem()
    variables["x"].setInitialValue(0)
    variables["y"].setInitialValue(1)

    with patch.object(highspy.Highs, "setSolution", autospec=True) as set_solution:
        prob.solve(_balance.WarmStartHiGHS(msg=False, gapRel=0))

    set_solution.assert_called_once()
    passed = set_solution.call_args.args[-1]
    assert list(passed.col_value) == [0, 1]
    # The warm start must not change the optimum.
    assert prob.status == pulp.LpStatusOptimal
    assert pulp.value(prob.objective) == pytest.approx(2.0)


def test_warm_start_solver_skips_without_values():
    """A first solve (no values yet) does not call setSolution."""
    prob, _ = _small_problem()
    with patch.object(highspy.Highs, "setSolution", autospec=True) as set_solution:
        prob.solve(_balance.WarmStartHiGHS(msg=False, gapRel=0))
    set_solution.assert_not_called()
    assert prob.status == pulp.LpStatusOptimal
