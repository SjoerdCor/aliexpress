"""Map a :class:`~.engine.Solution` into the shared :class:`SolutionResult`.

The reporting/export layer (``solver/solutions.py``) reads a
:class:`SolutionResult`, not solver-specific objects. This module is a pure
mapper: it derives every field the reporting layer needs from a solved
:class:`~.engine.Solution`, with no solving of its own.
"""

from dataclasses import dataclass

from ..data import preferences_data
from .engine import Solution


@dataclass(frozen=True)
class SexCounts:
    """Boys/girls counts for one jaarlaag cohort within a group."""

    boys: int
    girls: int


@dataclass(frozen=True)
class GroupComposition:
    """Boys/girls counts for one target group: total, and per jaarlaag cohort.

    ``per_year`` keys are jaarlaag numbers, or ``None`` for students without one (the
    Excel input path). ``boys_total``/``girls_total`` include the group's current
    occupancy; summing ``per_year`` gives the newly assigned students only.
    """

    boys_total: int
    girls_total: int
    per_year: dict[int | None, SexCounts]


@dataclass(frozen=True)
class SolutionResult:
    """Structured outcome of a solved distribution, read straight from the solver.

    Consumed by :class:`~aliexpress.solver.solutions.SolutionAnalyzer`. Every field
    holds plain Python values (no solver objects), so the result is straightforward
    to serialise once a persistence route is needed.

    The ``(student, Nr)`` keys index the positive ("Graag met") wishes: ``Nr`` is the
    wish's sequence number within that student's wishes; its target (a classmate or
    group) lives in ``preferences.loc[(student, "Graag met", Nr), "Waarde"]``.
    """

    assignment: dict[str, str]  # student -> assigned group
    student_satisfaction: dict[str, float]  # student -> relative satisfaction (0..1)
    satisfied: dict[tuple[str, int], bool]  # (student, Nr) -> wish fulfilled
    weighted_satisfied: dict[tuple[str, int], float]  # (student, Nr) -> weighted value
    weights: dict[tuple[str, int], float]  # (student, Nr) -> wish weight (signed)
    group_composition: dict[str, GroupComposition]  # group -> boys/girls counts


def to_solution_result(
    solution: Solution, preferences, students: dict, groups_to: dict
) -> SolutionResult:
    """Read a solved CP-SAT instance into a structured :class:`SolutionResult`.

    Parameters
    ----------
    solution : Solution
        The solved assignment, honored wishes and recomputed satisfaction.
    preferences : pandas.DataFrame
        Long-format preference rows, indexed by ``(student, TypeWens, Nr)``.
    students : dict
        Per-student info, keyed by student name (``Jongen/meisje`` is read here).
    groups_to : dict
        Target groups, keyed by group name, with current ``Jongens``/``Meisjes``
        occupancy.

    Returns
    -------
    SolutionResult
        The structured outcome the reporting layer consumes.
    """
    graag_met = preferences_data.get_graag_met(preferences)
    weights = dict(graag_met["Gewicht"])
    weighted_satisfied = {
        key: (s * weights[key] if weights[key] > 0 else (1 - s) * weights[key])
        for key, s in solution.satisfied.items()
    }
    return SolutionResult(
        assignment=solution.assignment,
        student_satisfaction=solution.student_satisfaction,
        satisfied=solution.satisfied,
        weighted_satisfied=weighted_satisfied,
        weights=weights,
        group_composition=_group_composition(solution.assignment, students, groups_to),
    )


def _group_composition(
    assignment: dict[str, str], students: dict, groups_to: dict
) -> dict[str, GroupComposition]:
    """Per target group, the boys/girls counts derived from the assignment, per jaarlaag."""
    composition = {}
    for group, occupancy in groups_to.items():
        per_year = _per_year_counts(assignment, students, group)
        composition[group] = GroupComposition(
            boys_total=occupancy["Jongens"] + sum(c.boys for c in per_year.values()),
            girls_total=occupancy["Meisjes"] + sum(c.girls for c in per_year.values()),
            per_year=per_year,
        )
    return composition


def _per_year_counts(
    assignment: dict[str, str], students: dict, group: str
) -> dict[int | None, SexCounts]:
    """Boys/girls counts for ``group``'s newly assigned students, by jaarlaag cohort."""
    tallies: dict[int | None, list[int]] = {}
    for student, assigned in assignment.items():
        if assigned != group:
            continue
        year = students[student].get("Jaarlaag")
        sex_index = 0 if students[student]["Jongen/meisje"] == "Jongen" else 1
        tallies.setdefault(year, [0, 0])[sex_index] += 1
    return {year: SexCounts(boys=b, girls=g) for year, (b, g) in tallies.items()}
