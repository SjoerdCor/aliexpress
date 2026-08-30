"""Behavioral tests for carrying a proven balance profile into a fresh model."""

from ortools.sat.python import cp_model

from aliexpress.solver import balance_profile
from aliexpress.solver._balance_families import FAMILY_NAMES, SLACK_WEIGHTS


class _SlackTupleCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, slacks):
        super().__init__()
        self.slacks = slacks
        self.rows = set()

    def on_solution_callback(self):
        """Record one allowed slack tuple returned by CP-SAT."""
        self.rows.add(tuple(self.Value(self.slacks[name]) for name in FAMILY_NAMES))


def test_exact_profile_keeps_every_valid_family_mapping():
    """Pin the sorted profile, not one incidental slack-to-family assignment.

    Four families weigh 100 and can therefore carry the profile's 200 and 100
    in any ordered pair; the two weight-49 families must stay zero. All twelve
    mappings remain feasible, and every one realizes the exact same profile.
    """
    model = cp_model.CpModel()
    slacks = {name: model.NewIntVar(0, 2, f"slack_{name}") for name in FAMILY_NAMES}
    expected_profile = (200, 100, 0, 0, 0, 0)

    balance_profile.pin_exact_profile(
        model,
        slacks,
        expected_profile,
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
        assert tuple(sorted(weighted, reverse=True)) == expected_profile
