"""Test for optimization strategies"""

# pylint: disable=redefined-outer-name, protected-access
from unittest.mock import patch

import pulp
import pytest

from aliexpress.errors import SolverError
from aliexpress.solver import optimizationstrategies


@pytest.fixture
def simple_scores():
    """Return a small dict of pulp variables with constraints."""
    prob = pulp.LpProblem("TestProb", pulp.LpMaximize)
    scores = {
        "a": pulp.LpVariable("a", lowBound=0, upBound=10),
        "b": pulp.LpVariable("b", lowBound=0, upBound=10),
        "c": pulp.LpVariable("c", lowBound=0, upBound=10),
    }
    return prob, scores


def test_total_returns_exact_sum(simple_scores):
    """Test total optimization strategy"""
    _, scores = simple_scores

    expr = optimizationstrategies.total(scores)

    assert isinstance(expr, pulp.LpAffineExpression)

    vars_in_expr = {var.name: coeff for var, coeff in expr.items()}
    assert vars_in_expr == {"a": 1.0, "b": 1.0, "c": 1.0}

    scores["a"].setInitialValue(2)
    scores["b"].setInitialValue(3)
    scores["c"].setInitialValue(5)
    value = pulp.value(expr)
    assert value == 10


def test_lowest_score_structure_and_constraints(simple_scores):
    """Check right constraints and outcome of lowest_score"""
    prob, scores = simple_scores

    expr = optimizationstrategies.lowest_score(scores, prob)
    assert isinstance(expr, pulp.LpAffineExpression)

    coeffs = {v.name: c for v, c in expr.items()}
    assert coeffs["MinimalScore"] == 1_000_000
    assert coeffs["a"] == 1.0
    assert coeffs["b"] == 1.0
    assert coeffs["c"] == 1.0

    constraint_strs = [str(c) for c in prob.constraints.values()]
    expected_constraints = [
        "MinimalScore - a <= 0",
        "MinimalScore - b <= 0",
        "MinimalScore - c <= 0",
    ]
    assert constraint_strs == expected_constraints


def test_lowest_score_evaluates_correctly():
    """Check lowest score evaluates correctly"""
    prob = pulp.LpProblem("EvalProb", pulp.LpMaximize)
    scores = {
        "a": pulp.LpVariable("a"),
        "b": pulp.LpVariable("b"),
        "c": pulp.LpVariable("c"),
    }

    expr = optimizationstrategies.lowest_score(scores, prob)
    prob += expr  # objective

    # Fixeer waarden zodat het oplosbaar is
    prob += scores["a"] == 2
    prob += scores["b"] == 5
    prob += scores["c"] == 7

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    min_val = min(2, 5, 7)
    total_val = 2 + 5 + 7
    expected_value = 1_000_000 * min_val + total_val

    assert pytest.approx(pulp.value(prob.objective)) == expected_value


def test_plateaud_lexmaxmin_two_variables_equal_distribution():
    """Test lexmaxmin optimizes for the lowest of variables"""
    prob = pulp.LpProblem("LexMaxMinLevel1", pulp.LpMaximize)
    a = pulp.LpVariable("a", lowBound=0, upBound=10)
    b = pulp.LpVariable("b", lowBound=0, upBound=10)
    a.setInitialValue(8)  # Force  lexmaxmin to equalize values
    b.setInitialValue(2)

    scores = {"a": a, "b": b}

    prob += a + b <= 10

    expr = optimizationstrategies.plateaud_lexmaxmin(
        scores, prob, n_levels_max=1, satisfaction_max=100
    )
    prob += expr

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    a_val = pulp.value(a)
    b_val = pulp.value(b)

    assert abs(a_val - b_val) < 1e-4
    assert abs(a_val + b_val - 10) < 1e-4


def test_lexmaxmin_three_variables_equal_distribution():
    """Test lexmaxmin equalizes multiple variables correctly"""
    prob = pulp.LpProblem("LexMaxMinLevel2", pulp.LpMaximize)
    a = pulp.LpVariable("a", lowBound=0, upBound=1)
    b = pulp.LpVariable("b", lowBound=0, upBound=1)
    c = pulp.LpVariable("c", lowBound=0, upBound=1)
    a.setInitialValue(0.01)
    c.setInitialValue(0.99)
    scores = {"a": a, "b": b, "c": c}

    prob += a + b + c <= 1.5

    expr = optimizationstrategies.plateaud_lexmaxmin(scores, prob)
    prob += expr

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    values = [pulp.value(v) for v in scores.values()]
    min_val = min(values)
    max_val = max(values)
    assert max_val - min_val < 1e-4
    assert abs(sum(values) - 1.5) < 1e-4, abs(sum(values) - 1.5)

    prob = pulp.LpProblem("LexMaxMinLevel4", pulp.LpMaximize)
    a = pulp.LpVariable("a", lowBound=0, upBound=10)
    b = pulp.LpVariable("b", lowBound=0, upBound=10)
    c = pulp.LpVariable("c", lowBound=0, upBound=10)
    scores = {"a": a, "b": b, "c": c}

    prob += a + b + c <= 15
    prob += a >= 2
    prob += b >= 4
    prob += c >= 3

    with patch(
        "aliexpress.solver.pulp_thresholds.apply_threshold_constraints"
    ) as mock_apply:
        mock_apply.side_effect = lambda *a_, **k: None

        expr = optimizationstrategies.plateaud_lexmaxmin(
            scores, prob, n_levels_max=3, satisfaction_max=100
        )
        prob += expr
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    values = [pulp.value(v) for v in scores.values()]
    min_val = min(values)
    # De kleinste waarde moet gemaximaliseerd zijn
    assert abs(min_val - 5) < 1e-3
    assert abs(sum(values) - 15) < 1e-3


def test_lexmaxmin_raises_clear_error_on_first_non_optimal_solve():
    """Non-optimal first sub-solve raises SolverError (not TypeError) via plateaud_lexmaxmin."""
    prob = pulp.LpProblem("TestNonOptimal", pulp.LpMaximize)
    scores = {
        "a": pulp.LpVariable("a", lowBound=0, upBound=10),
        "b": pulp.LpVariable("b", lowBound=0, upBound=10),
    }
    prob += scores["a"] + scores["b"] <= 10

    def infeasible_solve(*_args, **_kwargs):
        prob.status = -1  # pulp.LpStatusInfeasible
        return -1

    with patch.object(prob, "solve", infeasible_solve):
        with pytest.raises(SolverError, match="level 0.*Infeasible"):
            optimizationstrategies.plateaud_lexmaxmin(scores, prob)


def test_lexmaxmin_raises_clear_error_on_second_non_optimal_solve():
    """Non-optimal second sub-solve (count phase) raises SolverError via plateaud_lexmaxmin."""
    prob = pulp.LpProblem("TestNonOptimal2", pulp.LpMaximize)
    scores = {
        "a": pulp.LpVariable("a", lowBound=0, upBound=10),
        "b": pulp.LpVariable("b", lowBound=0, upBound=10),
    }
    prob += scores["a"] + scores["b"] <= 10

    original_solve = prob.solve
    call_count = [0]

    def mock_solve(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return original_solve(
                *args, **kwargs
            )  # first call: real solve to get past _raise_minimal_score
        prob.status = -1  # Infeasible on the second call (count phase)
        return -1

    with patch.object(prob, "solve", mock_solve):
        with pytest.raises(SolverError, match="Infeasible"):
            optimizationstrategies.plateaud_lexmaxmin(scores, prob)


def test_lexmaxmin_big_m_derived_from_score_bounds():
    """The big-M spans the scores' bound range instead of a fixed constant."""
    prob = pulp.LpProblem("BigM", pulp.LpMaximize)
    scores = {
        "a": pulp.LpVariable("a", lowBound=-2, upBound=1),
        "b": pulp.LpVariable("b", lowBound=0, upBound=1),
    }
    solver = optimizationstrategies._PlateaudLexMaxMin(scores, prob, None)
    # Range [-2, 1] -> spread 3, plus 1 margin.
    assert solver.big_m == pytest.approx(4.0)


def test_lexmaxmin_big_m_falls_back_when_unbounded():
    """Scores without bounds fall back to the conservative class constant."""
    prob = pulp.LpProblem("BigMFallback", pulp.LpMaximize)
    scores = {"a": pulp.LpVariable("a")}
    solver = optimizationstrategies._PlateaudLexMaxMin(scores, prob, None)
    assert solver.big_m == optimizationstrategies._PlateaudLexMaxMin.BIG_M_FALLBACK


def test_lexmaxmin_multiple_plateaus():
    """Test that lexmaxmin finds different plateaus if needed"""
    prob = pulp.LpProblem("LexMaxMinLevel3", pulp.LpMaximize)
    a = pulp.LpVariable("a", lowBound=0, upBound=10)
    b = pulp.LpVariable("b", lowBound=0, upBound=10)
    c = pulp.LpVariable("c", lowBound=0, upBound=10)
    scores = {"a": a, "b": b, "c": c}

    prob += a + b <= 5
    prob += a + c <= 6
    prob += b + c <= 7

    expr = optimizationstrategies.plateaud_lexmaxmin(scores, prob)
    prob += expr
    print(prob)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    a_val = pulp.value(a)
    b_val = pulp.value(b)
    c_val = pulp.value(c)

    assert a_val == pytest.approx(2.5, abs=1e-4)
    assert b_val == pytest.approx(2.5, abs=1e-4)
    assert c_val == pytest.approx(3.5, abs=1e-4)
