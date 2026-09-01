"""Behavioral tests for carrying sorted weighted slacks into a fresh model."""

import pytest
from ortools.sat.python import cp_model

from aliexpress.solver import sorted_weighted_slacks
from aliexpress.solver._balance_families import FAMILY_NAMES, SLACK_WEIGHTS
from aliexpress.solver.engine import _sorting_network_descending


class _SlackTupleCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, slacks):
        super().__init__()
        self.slacks = slacks
        self.rows = set()

    def on_solution_callback(self):
        """Record one allowed slack tuple returned by CP-SAT."""
        self.rows.add(tuple(self.Value(self.slacks[name]) for name in FAMILY_NAMES))


def test_exact_sorted_weighted_slacks_keeps_every_valid_family_mapping():
    """Pin sorted weighted slacks, not one incidental family assignment.

    Four families weigh 100 and can therefore carry the values 200 and 100 in
    any ordered pair; the two weight-49 families must stay zero. All twelve
    mappings remain feasible, and every one realizes the same sorted values.
    """
    model = cp_model.CpModel()
    slacks = {name: model.NewIntVar(0, 2, f"slack_{name}") for name in FAMILY_NAMES}
    expected_sorted_weighted_slacks = (200, 100, 0, 0, 0, 0)

    sorted_weighted_slacks.pin_exact_sorted_weighted_slacks(
        model,
        slacks,
        expected_sorted_weighted_slacks,
        slack_upper_bounds={name: 2 for name in FAMILY_NAMES},
    )

    collector = _SlackTupleCollector(slacks)
    solver = cp_model.CpSolver()
    solver.SearchForAllSolutions(model, collector)

    assert len(collector.rows) == 12
    for row in collector.rows:
        weighted = [
            SLACK_WEIGHTS[name] * value for name, value in zip(FAMILY_NAMES, row)
        ]
        assert tuple(sorted(weighted, reverse=True)) == expected_sorted_weighted_slacks


@pytest.mark.parametrize(
    "invalid_sorted_weighted_slacks, upper_bound",
    [
        ((1, 0, 0, 0, 0, 0), 2),
        ((200, 100, 0, 0, 0, 0), 1),
    ],
)
def test_rejects_invalid_sorted_weighted_slacks_table_values(
    invalid_sorted_weighted_slacks, upper_bound
):
    """Reject values that no family can realize within its slack domain."""
    model = cp_model.CpModel()
    slacks = {
        name: model.NewIntVar(0, upper_bound, f"slack_{name}") for name in FAMILY_NAMES
    }

    with pytest.raises(ValueError, match="no valid slack tuple"):
        sorted_weighted_slacks.pin_exact_sorted_weighted_slacks(
            model,
            slacks,
            invalid_sorted_weighted_slacks,
            slack_upper_bounds={name: upper_bound for name in FAMILY_NAMES},
        )


def test_sorting_network_orders_values_descending():
    """The compare-swap network exposes the input values in descending order."""
    model = cp_model.CpModel()
    inputs = [model.NewIntVar(0, 4, f"input_{i}") for i in range(4)]
    for variable, value in zip(inputs, (3, 1, 4, 2)):
        model.Add(variable == value)

    outputs = _sorting_network_descending(model, inputs, upper_bound=4)

    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    assert [solver.Value(output) for output in outputs] == [4, 3, 2, 1]
