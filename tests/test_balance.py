"""Tests for _balance.py: solver construction and warm starts."""

# pylint: disable=protected-access

import os
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


def test_partial_start_none_without_any_values():
    """Without previous values there is nothing to warm-start from."""
    prob, _ = _small_problem()
    for i, var in enumerate(prob.variables()):
        var.index = i
    assert _balance._partial_start(prob) is None


def test_partial_start_covers_only_valued_variables():
    """Only variables carrying a value appear, as (column index, value) pairs."""
    prob, variables = _small_problem()
    y_column = None
    for i, var in enumerate(prob.variables()):
        var.index = i
        if var.name == "y":
            y_column = i
    variables["y"].setInitialValue(1)
    indices, values = _balance._partial_start(prob)
    assert list(indices) == [y_column]
    assert list(values) == [1.0]


def test_warm_start_solver_passes_solution_to_highs():
    """WarmStartHiGHS hands the current values to HiGHS as a partial MIP start."""
    prob, variables = _small_problem()
    variables["x"].setInitialValue(0)
    variables["y"].setInitialValue(1)

    with patch.object(highspy.Highs, "setSolution", autospec=True) as set_solution:
        prob.solve(_balance.WarmStartHiGHS(msg=False, gapRel=0))

    set_solution.assert_called_once()
    n_entries, indices, values = set_solution.call_args.args[-3:]
    assert n_entries == 2
    assert len(indices) == 2
    assert sorted(values) == [0.0, 1.0]
    # The warm start must not change the optimum.
    assert prob.status == pulp.LpStatusOptimal
    assert pulp.value(prob.objective) == pytest.approx(2.0)


def test_get_solver_uses_warm_start_all_cores_and_heuristics():
    """get_solver returns the warm-start solver, uses every core and raised
    heuristic effort: the primal (incumbent) side is the measured bottleneck."""
    solver = _balance.get_solver()
    assert isinstance(solver, _balance.WarmStartHiGHS)
    assert solver.threads == os.cpu_count()
    assert solver.optionsDict["mip_heuristic_effort"] == pytest.approx(0.25)


def test_warm_start_solver_defaults_to_all_cores():
    """Every WarmStartHiGHS asks for all cores, whatever the construction site.

    HiGHS initializes its global scheduler at the first parallel solve in the
    process; a later solve asking for more threads than that first one returns
    'Not Solved'.  The class-level default makes the first ask the largest, so
    solve order cannot break (regression for the mixed-thread-count failure)."""
    assert _balance.WarmStartHiGHS(msg=False).threads == os.cpu_count()
    prob, variables = _small_problem()
    variables["x"].setInitialValue(0)
    variables["y"].setInitialValue(1)
    prob.solve(_balance.WarmStartHiGHS(msg=False, gapRel=0))
    prob2, _ = _small_problem()
    prob2.solve(_balance.get_solver())
    assert prob.status == pulp.LpStatusOptimal
    assert prob2.status == pulp.LpStatusOptimal


def test_warm_start_solver_skips_without_values():
    """A first solve (no values yet) does not call setSolution."""
    prob, _ = _small_problem()
    with patch.object(highspy.Highs, "setSolution", autospec=True) as set_solution:
        prob.solve(_balance.WarmStartHiGHS(msg=False, gapRel=0))
    set_solution.assert_not_called()
    assert prob.status == pulp.LpStatusOptimal
