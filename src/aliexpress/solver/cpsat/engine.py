"""The CP-SAT solve pipeline: build the model, run it, extract the solution.

Orchestrates the two entry points below: build the constraints via
:mod:`.model`, fix any lexicographic pre-stages a path needs (the automatic
path's minimal-relaxation search), hand off to :mod:`.strategies` for the
chosen aggregate objective, and extract the proven-optimal solution into a
plain :class:`CpSatSolution`.

The reported per-student satisfaction is *recomputed in float* from the honored
wishes — not read back as ``integer / SATISFACTION_SCALE`` — so the ×10^6
rounding can never leak into the report and the pinned integration values stay
exact.
"""

from dataclasses import dataclass

from ortools.sat.python import cp_model

from ...data import preferences_data
from ..satisfaction import get_satisfaction_integral
from . import model as cpsat_model
from . import strategies
from ._balance_families import SLACK_WEIGHTS


@dataclass
class CpSatSolution:
    """Solved outcome, in plain Python values (no solver objects)."""

    assignment: dict  # student -> group
    satisfied: dict  # (student, Nr) -> bool (wish honored)
    student_satisfaction: dict  # student -> float, recomputed from honored wishes


def solve_with_fixed_balance(  # pylint: disable=too-many-arguments
    # Each keyword-only argument is a distinct input to the model (raw data,
    # rules, balance limits, strategy choice); grouping them would obscure the
    # entry point's public interface rather than simplify it.
    *,
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
    groupbalance,
    optimize: str = "lexmaxmin",
) -> CpSatSolution:
    """Solve the distribution with hard balance limits (the manual path).

    Builds the model via :func:`.model.build_problem`, runs the chosen
    optimization strategy, and returns the solved values.

    Parameters
    ----------
    preferences : pandas.DataFrame
        Long-format preference rows, indexed by ``(student, TypeWens, Nr)``.
    students : dict
        Per-student info (``Jaarlaag``, ``Jongen/meisje``, ``Stamgroep``,
        ``MinimaleTevredenheid``).
    groups_to : dict
        Target groups, keyed by group name, with current ``Jongens``/``Meisjes``
        occupancy.
    not_together : list
        Rules of the form ``{"group": {student, ...}, "Max_aantal_samen": int}``.
    groupbalance : aliexpress.solver._balance.GroupBalance
        The hard limit for each of the six class-balance families.
    optimize : str, optional
        Which aggregate objective to optimize: ``"lexmaxmin"`` (default,
        plateaud lexicographic max-min with a total-satisfaction tie-break) or
        ``"total"`` (maximize the total satisfaction directly). See
        :mod:`.strategies` for the trade-off between the two.

    Returns
    -------
    CpSatSolution
        The solved assignment, honored wishes and recomputed satisfaction.

    Raises
    ------
    SolverError
        If any stage cannot be solved to proven optimality.
    """
    problem = cpsat_model.build_problem(
        preferences, students, groups_to, not_together, groupbalance
    )
    solver = strategies.optimize(problem, optimize)
    return _extract(problem, solver, preferences)


def solve_within_minimal_relaxation(
    *,
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
    optimize: str = "lexmaxmin",
) -> CpSatSolution:
    """Solve the distribution with the class balance relaxed only as far as needed.

    Builds the model via :func:`.model.build_soft_problem` and fixes the class
    balance in two lexicographic stages before the main solve:

    1. Minimize the number of students left without any honored positive wish
       (normally 0), then pin that count as an upper bound. A student cannot
       keep a positive wish if the balance that would give it to them is
       forbidden, so this stage finds how much relaxation is unavoidable.
    2. With that count pinned, minimize the weighted balance relaxation, then
       pin the resulting minimum too. Whole-group limits weigh less than
       per-year ones (:data:`~._balance_families.SLACK_WEIGHTS`), and a
       max-slack term keeps the relaxation spread across limits rather than
       piled onto one.

    The chosen strategy then runs on the same model, now that the class
    balance is fixed at its minimal relaxation.

    Parameters
    ----------
    preferences : pandas.DataFrame
        Long-format preference rows, indexed by ``(student, TypeWens, Nr)``.
    students : dict
        Per-student info (``Jaarlaag``, ``Jongen/meisje``, ``Stamgroep``,
        ``MinimaleTevredenheid``).
    groups_to : dict
        Target groups, keyed by group name, with current ``Jongens``/``Meisjes``
        occupancy.
    not_together : list
        Rules of the form ``{"group": {student, ...}, "Max_aantal_samen": int}``.
    optimize : str, optional
        Which aggregate objective to optimize: ``"lexmaxmin"`` (default,
        plateaud lexicographic max-min with a total-satisfaction tie-break) or
        ``"total"`` (maximize the total satisfaction directly). See
        :mod:`.strategies` for the trade-off between the two.

    Returns
    -------
    CpSatSolution
        The solved assignment, honored wishes and recomputed satisfaction.

    Raises
    ------
    SolverError
        If any stage cannot be solved to proven optimality.
    """
    problem = cpsat_model.build_soft_problem(
        preferences, students, groups_to, not_together
    )
    model = problem.model

    solver = strategies.solve_stage(
        model, "unmet wishes", minimize=sum(problem.unmet.values())
    )
    unmet_optimum = round(solver.ObjectiveValue())
    model.Add(sum(problem.unmet.values()) <= unmet_optimum)

    max_slack = model.NewIntVar(0, len(students), "max_slack")
    model.AddMaxEquality(max_slack, list(problem.slacks.values()))
    weighted = (
        sum(SLACK_WEIGHTS[name] * slack for name, slack in problem.slacks.items())
        + 100 * max_slack
    )
    solver = strategies.solve_stage(model, "balance relaxation", minimize=weighted)
    budget = round(solver.ObjectiveValue())
    model.Add(weighted <= budget)

    solver = strategies.optimize(problem, optimize)
    return _extract(problem, solver, preferences)


def _extract(problem, solver: cp_model.CpSolver, preferences) -> CpSatSolution:
    """Read the solved values; satisfaction is recomputed in float per student.

    Parameters
    ----------
    problem : model.CpSatProblem | model.CpSatSoftProblem
        The built model, for the ``in_group``/``satisfied``/``satisfaction``
        variables to read back.
    solver : cp_model.CpSolver
        The solver holding the final stage's proven-optimal solution.
    preferences : pandas.DataFrame
        Long-format preference rows, indexed by ``(student, TypeWens, Nr)``.

    Returns
    -------
    CpSatSolution
        The solved assignment, honored wishes and recomputed satisfaction.
    """
    assignment = {
        student: group
        for (student, group), var in problem.in_group.items()
        if solver.BooleanValue(var)
    }
    satisfied = {
        key: solver.BooleanValue(literal) for key, literal in problem.satisfied.items()
    }
    return CpSatSolution(
        assignment=assignment,
        satisfied=satisfied,
        student_satisfaction=_float_satisfaction(
            preferences, satisfied, list(problem.satisfaction)
        ),
    )


def _float_satisfaction(preferences, satisfied: dict, students: list) -> dict:
    """Per-student float satisfaction from the honored wishes.

    The float twin of the model's integer element table: F(weighted honored
    sum), normalized by F(best case) when the student has positive wishes, or
    added to the baseline 1 when not.

    Parameters
    ----------
    preferences : pandas.DataFrame
        Long-format preference rows, indexed by ``(student, TypeWens, Nr)``.
    satisfied : dict
        Honored boolean per ``(student, Nr)`` preference row.
    students : list
        The students to report a satisfaction value for.

    Returns
    -------
    dict[str, float]
        Per-student satisfaction, exact to the ×10^6 integer scale.
    """
    graag_met = preferences_data.get_graag_met(preferences)
    honored_sum: dict[str, float] = {}
    best_sum: dict[str, float] = {}
    for key, row in graag_met.iterrows():
        student, weight = key[0], row["Gewicht"]
        honored = satisfied[key]
        # Honored positive wish: +weight. Violated negative wish: its (negative)
        # weight. Otherwise 0 — identical to the model's weighted sum.
        contribution = weight if (weight > 0) == honored else 0.0
        honored_sum[student] = honored_sum.get(student, 0.0) + contribution
        best_sum[student] = best_sum.get(student, 0.0) + max(weight, 0.0)

    result = {}
    for student in students:
        if student not in honored_sum:
            result[student] = 1.0
            continue
        raw = get_satisfaction_integral(0, honored_sum[student])
        if best_sum[student] > 0:
            result[student] = raw / get_satisfaction_integral(0, best_sum[student])
        else:
            result[student] = 1.0 + raw
    return result
