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

from ortools.sat.python import cp_model

from ._balance_families import SLACK_WEIGHTS


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
