"""Shared primitives for the solver sub-package: ``GroupBalance`` (fixed limits),
``BalanceMaxima`` (per-family upper bounds on the automatic relaxation) and its
data-driven default factory."""

import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class GroupBalance:
    """
    Constraints controlling how students are distributed across groups.

    All values must be non-negative integers.
    """

    max_clique: int = 5
    """The number of students that can go to the same group"""

    max_clique_sex: int = 3
    """Maximum number of students of the same sex from the same original group in a group."""

    max_diff_n_students_year: int = 2
    """Max difference between largest and smallest group per year."""

    max_diff_n_students_total: int = 3
    """Max difference between largest and smallest group overall."""

    max_imbalance_boys_girls_year: int = 2
    """Max difference between boys and girls per year in a group."""

    max_imbalance_boys_girls_total: int = 3
    """Max difference between boys and girls in total per group."""

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


# All six balance limits set to 1: the tightest possible balance.
# Used as the starting point for the adaptive relaxation search.
STRICTEST_BALANCE = GroupBalance(1, 1, 1, 1, 1, 1)


@dataclass(frozen=True)
class BalanceMaxima:
    """Per-family upper bound on the automatic relaxation. None = unlimited."""

    max_diff_n_students_year: int | None = None
    """Upper bound on max_diff_n_students_year during the adaptive relaxation."""

    max_diff_n_students_total: int | None = None
    """Upper bound on max_diff_n_students_total during the adaptive relaxation."""

    max_imbalance_boys_girls_year: int | None = None
    """Upper bound on max_imbalance_boys_girls_year during the adaptive relaxation."""

    max_imbalance_boys_girls_total: int | None = None
    """Upper bound on max_imbalance_boys_girls_total during the adaptive relaxation."""

    max_clique: int | None = None
    """Upper bound on max_clique during the adaptive relaxation."""

    max_clique_sex: int | None = None
    """Upper bound on max_clique_sex during the adaptive relaxation."""

    def __post_init__(self):
        for name, value in vars(self).items():
            if value is None:
                continue
            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer or None, got {type(value).__name__}"
                )
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

    def constrains_anything(self) -> bool:
        """True if at least one family has a cap (a non-None field)."""
        return any(value is not None for value in vars(self).values())


#: The null-object "no caps" BalanceMaxima: every family unlimited. It is the
#: default for every solve entry point, so the solver may always assume it
#: receives a BalanceMaxima (never None). Safe to share as a default argument
#: because BalanceMaxima is frozen.
UNCAPPED = BalanceMaxima()


#: Generous floor for the whole-group defaults. The ceiling only exists to stop
#: the automatic relaxation from running away, not to bind a healthy instance,
#: so it sits well above what a normal spread needs — the relaxation should
#: almost never reach it.
_WHOLE_GROUP_FLOOR = 4

#: Fixed ceiling for the per-cohort defaults. A single year cohort is smaller
#: and more visible than the whole group, so its imbalance is capped tighter —
#: but still generously enough that the relaxation almost never needs it.
_PER_YEAR_DEFAULT = 3


def default_balance_maxima(students: dict, groups_to: dict) -> BalanceMaxima:
    """Data-driven default ceilings for the automatic relaxation.

    Each ceiling is deliberately generous: it exists only to stop the automatic
    relaxation from running away, never to bind a healthy instance. The values
    are derived from the instance so they are never *stricter* than what the
    current occupancy or the group structure already forces.

    - ``max_diff_n_students_total`` / ``max_imbalance_boys_girls_total``: the
      larger of a generous floor (:data:`_WHOLE_GROUP_FLOOR`) and the spread the
      *current* occupancy already exhibits. Capping below the occupancy's own
      spread would be infeasible before a single student is placed, so the floor
      only ever raises the ceiling, never lowers it.
    - ``max_diff_n_students_year`` / ``max_imbalance_boys_girls_year``: a fixed,
      tighter :data:`_PER_YEAR_DEFAULT` — a single cohort is smaller and its
      imbalance more visible than the whole group's.
    - ``max_clique`` / ``max_clique_sex``: twice the even-split floor. The
      smallest possible clique per target group is the largest original group
      (or its largest same-sex part) spread evenly over the groups,
      ``ceil(size / n_groups)``. Doubling that leaves the relaxation room to keep
      a few extra classmates together when honouring wishes needs it, without
      letting a whole original group pile into one target group.

    Parameters
    ----------
    students : dict
        Per-student info, keyed by student, with ``Stamgroep`` and
        ``Jongen/meisje``.
    groups_to : dict
        Target groups, keyed by group name, with current ``Jongens``/``Meisjes``
        occupancy.

    Returns
    -------
    BalanceMaxima
        The six default ceilings for this instance.
    """
    n_groups = len(groups_to)
    totals = [counts["Jongens"] + counts["Meisjes"] for counts in groups_to.values()]
    occupancy_spread = max(totals) - min(totals)
    occupancy_imbalance = max(
        abs(counts["Jongens"] - counts["Meisjes"]) for counts in groups_to.values()
    )

    largest_clique = max(Counter(s["Stamgroep"] for s in students.values()).values())
    largest_same_sex = max(
        Counter(
            (s["Stamgroep"], s["Jongen/meisje"]) for s in students.values()
        ).values()
    )

    return BalanceMaxima(
        max_diff_n_students_total=max(_WHOLE_GROUP_FLOOR, occupancy_spread),
        max_imbalance_boys_girls_total=max(_WHOLE_GROUP_FLOOR, occupancy_imbalance),
        max_diff_n_students_year=_PER_YEAR_DEFAULT,
        max_imbalance_boys_girls_year=_PER_YEAR_DEFAULT,
        max_clique=2 * math.ceil(largest_clique / n_groups),
        max_clique_sex=2 * math.ceil(largest_same_sex / n_groups),
    )
