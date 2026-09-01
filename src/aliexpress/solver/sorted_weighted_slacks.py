"""Carry proven sorted weighted slacks between independent CP-SAT models.

The balance phase proves only the sorted multiset of ``weight * slack``
values. It deliberately does not choose which family owns which position in
those sorted weighted slacks. A fresh satisfaction model must preserve that
freedom: pinning the six family slacks from one incidental balance assignment
would make the model stricter than the proven leximin optimum.

This module boundary is deliberate: :mod:`._balance_families` defines balance
inside one model, while :mod:`.engine` orchestrates solve stages. This module
owns the translation of one stage's proven sorted weighted slacks into
constraints on a new model, independently testable without either orchestration
or UI concerns.
"""

import itertools
from dataclasses import dataclass

from ortools.sat.python import cp_model

from ._balance_families import SLACK_WEIGHTS


@dataclass(frozen=True)
class LeximinOutcome:
    """The solver and sorted values proven by a generic descending leximin run."""

    solver: cp_model.CpSolver
    values: tuple[int, ...]


def sorting_network_descending(
    model: cp_model.CpModel,
    values: list,
    upper_bound: int,
    *,
    variable_prefix: str = "balance_sort",
) -> list[cp_model.IntVar]:
    """Materialize ``values`` in descending order with exact compare-swaps.

    This is shared by the normal balance optimization and the cap-overflow
    diagnosis. Both callers need the same order statistics, but keep their own
    model-building responsibilities and objective meaning.
    """
    ordered = []
    for input_index, value in enumerate(values):
        current = value
        next_ordered = []
        for position, existing in enumerate(ordered):
            high = model.NewIntVar(
                0,
                upper_bound,
                f"{variable_prefix}_{input_index}_{position}_high",
            )
            low = model.NewIntVar(
                0,
                upper_bound,
                f"{variable_prefix}_{input_index}_{position}_low",
            )
            model.AddMaxEquality(high, [existing, current])
            model.AddMinEquality(low, [existing, current])
            next_ordered.append(high)
            current = low
        next_ordered.append(current)
        ordered = next_ordered
    return ordered


# Six explicit parameters keep the reusable helper's model, objective values,
# stage runner, and variable naming visible to both production callers.
def minimize_sorted_leximin(  # pylint: disable=too-many-arguments
    model: cp_model.CpModel,
    values: list,
    upper_bound: int,
    *,
    solve_stage,
    label_prefix: str,
    variable_prefix: str,
) -> LeximinOutcome:
    """Minimize non-negative ``values`` leximin after sorting them descending.

    ``solve_stage`` is injected so this reusable model helper does not own the
    solver orchestration. Each sorted position is proven and pinned in turn;
    once the largest remaining value is zero, all later values are zero too.
    """
    ordered = sorting_network_descending(
        model, values, upper_bound, variable_prefix=variable_prefix
    )
    solver = None
    proven_values = []
    for position, sorted_value in enumerate(ordered):
        solver = solve_stage(
            model, f"{label_prefix} M_{position}", minimize=sorted_value
        )
        value = round(solver.ObjectiveValue())
        proven_values.append(value)
        model.Add(sorted_value <= value)
        if value == 0:
            break
    proven_values.extend([0] * (len(ordered) - len(proven_values)))
    return LeximinOutcome(solver, tuple(proven_values))


def pin_exact_sorted_weighted_slacks(
    model: cp_model.CpModel,
    slacks: dict,
    sorted_weighted_slacks: tuple[int, ...],
    slack_upper_bounds: dict[str, int],
) -> None:
    """Require ``slacks`` to realize exactly ``sorted_weighted_slacks`` when sorted.

    At most ``6!`` permutations exist. Rows that cannot belong to a family
    because of its weight or explicit upper bound are discarded, and duplicate
    rows collapse. ``AddAllowedAssignments`` then gives the fresh model a
    compact, strongly propagating representation of the exact sorted weighted
    slacks without fixing one arbitrary slack-to-family mapping. Bounds are
    passed as plain integers because reflecting an OR-Tools variable domain can
    crash the Windows Python binding for constant-domain variables.
    """
    names = tuple(slacks)
    rows = set()
    for permutation in set(itertools.permutations(sorted_weighted_slacks)):
        row = []
        for name, weighted_value in zip(names, permutation):
            weight = SLACK_WEIGHTS[name]
            if weighted_value % weight:
                break
            slack_value = weighted_value // weight
            if not 0 <= slack_value <= slack_upper_bounds[name]:
                break
            row.append(slack_value)
        else:
            rows.add(tuple(row))

    if not rows:
        raise ValueError(
            "sorted weighted slacks "
            f"{sorted_weighted_slacks!r} have no valid slack tuple"
        )
    model.AddAllowedAssignments([slacks[name] for name in names], sorted(rows))
