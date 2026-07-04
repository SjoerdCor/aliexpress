"""Map a :class:`~.engine.CpSatSolution` into the shared :class:`SolutionResult`.

The reporting/export layer (``solver/solutions.py``) reads a
:class:`~aliexpress.solver.problemsolver.SolutionResult`, not solver-specific
objects. This module is a pure mapper: it derives every field the reporting
layer needs from a solved :class:`~.engine.CpSatSolution`, with no solving of
its own.
"""

from ...data import preferences_data
from ..problemsolver import GroupComposition, SolutionResult
from .engine import CpSatSolution


def to_solution_result(
    solution: CpSatSolution, preferences, students: dict, groups_to: dict
) -> SolutionResult:
    """Read a solved CP-SAT instance into a structured :class:`SolutionResult`.

    Parameters
    ----------
    solution : CpSatSolution
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
    """Per target group, the boys/girls counts derived from the assignment."""
    composition = {}
    for group, occupancy in groups_to.items():
        boys_year = sum(
            1
            for student, assigned in assignment.items()
            if assigned == group and students[student]["Jongen/meisje"] == "Jongen"
        )
        girls_year = sum(
            1
            for student, assigned in assignment.items()
            if assigned == group and students[student]["Jongen/meisje"] == "Meisje"
        )
        composition[group] = GroupComposition(
            boys_total=occupancy["Jongens"] + boys_year,
            girls_total=occupancy["Meisjes"] + girls_year,
            boys_year=boys_year,
            girls_year=girls_year,
        )
    return composition
