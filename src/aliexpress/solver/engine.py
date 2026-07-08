"""The CP-SAT solve pipeline: build the model, run it, extract the solution.

Orchestrates the two entry points below: build the constraints via
:mod:`.modelbuilder`, fix any lexicographic pre-stages a path needs (the
automatic path's minimal-relaxation search), hand off to :mod:`.strategies`
for the chosen aggregate objective, and extract the proven-optimal solution
into a plain :class:`Solution`.

The reported per-student satisfaction is *recomputed in float* from the honored
wishes — not read back as ``integer / SATISFACTION_SCALE`` — so the ×10^6
rounding can never leak into the report and the pinned integration values stay
exact.
"""

from dataclasses import dataclass

from ortools.sat.python import cp_model

from .. import errors
from ..data import preferences_data
from . import feasibility, modelbuilder, strategies
from ._balance_families import SLACK_WEIGHTS, max_slack_bound
from .satisfaction import _normalize_and_bound

#: Weight of the max-slack spreading term in the relaxation objective below:
#: equal to weight 1.0 on the ×100 scale :data:`~._balance_families.SLACK_WEIGHTS`
#: uses, so the max-slack term weighs exactly as much as one per-year family —
#: enough to break ties towards spreading the relaxation across limits rather
#: than piling it onto one, without dominating the per-family weights.
MAX_SLACK_WEIGHT = 100


@dataclass
class Solution:
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
) -> Solution:
    """Solve the distribution with hard balance limits (the manual path).

    Builds the model via :func:`.modelbuilder.build_problem`, runs the chosen
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
    Solution
        The solved assignment, honored wishes and recomputed satisfaction.

    Raises
    ------
    SolverError
        If any stage cannot be solved to proven optimality.
    """
    problem = modelbuilder.build_problem(
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
) -> Solution:
    """Solve the distribution with the class balance relaxed only as far as needed.

    Builds the model via :func:`.modelbuilder.build_soft_problem` and fixes the class
    balance in two lexicographic stages before the main solve:

    1. Minimize the number of students left at or below zero satisfaction
       (normally 0), then pin that count as an upper bound. A student cannot
       reach strictly positive satisfaction if the balance that would give
       them one is forbidden, so this stage finds how much relaxation is
       unavoidable.
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
    Solution
        The solved assignment, honored wishes and recomputed satisfaction.

    Raises
    ------
    FeasibilityError
        If the hard preference constraints (minimal satisfaction, not-together,
        "Niet in") are mutually infeasible even with class balance fully soft —
        detected by the first stage below coming back ``INFEASIBLE``. The
        ``context["case"]`` names the diagnosed cause (see
        :func:`.feasibility.diagnose`).
    SolverError
        If any other stage cannot be solved to proven optimality.
    """
    problem = modelbuilder.build_soft_problem(
        preferences, students, groups_to, not_together
    )
    model = problem.model

    try:
        solver = strategies.solve_stage(
            model,
            "non-positive satisfaction",
            minimize=sum(problem.nonpositive.values()),
        )
    except errors.StageInfeasible as exc:
        raise errors.FeasibilityError(
            "infeasible_preferences",
            context={
                "case": feasibility.diagnose(
                    preferences=preferences,
                    students=students,
                    groups_to=groups_to,
                    not_together=not_together,
                )
            },
            technical_message="Hard preference constraints are mutually infeasible",
        ) from exc
    nonpositive_optimum = round(solver.ObjectiveValue())
    model.Add(sum(problem.nonpositive.values()) <= nonpositive_optimum)

    max_slack = model.NewIntVar(0, max_slack_bound(students, groups_to), "max_slack")
    model.AddMaxEquality(max_slack, list(problem.slacks.values()))
    weighted = (
        sum(SLACK_WEIGHTS[name] * slack for name, slack in problem.slacks.items())
        + MAX_SLACK_WEIGHT * max_slack
    )
    solver = strategies.solve_stage(model, "balance relaxation", minimize=weighted)
    budget = round(solver.ObjectiveValue())
    model.Add(weighted <= budget)

    solver = strategies.optimize(problem, optimize)
    return _extract(problem, solver, preferences)


def _extract(problem, solver: cp_model.CpSolver, preferences) -> Solution:
    """Read the solved values; satisfaction is recomputed in float per student.

    Parameters
    ----------
    problem : modelbuilder.Problem | modelbuilder.SoftProblem
        The built model, for the ``in_group``/``satisfied``/``satisfaction``
        variables to read back.
    solver : cp_model.CpSolver
        The solver holding the final stage's proven-optimal solution.
    preferences : pandas.DataFrame
        Long-format preference rows, indexed by ``(student, TypeWens, Nr)``.

    Returns
    -------
    Solution
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
    return Solution(
        assignment=assignment,
        satisfied=satisfied,
        student_satisfaction=_float_satisfaction(
            preferences, satisfied, list(problem.satisfaction)
        ),
    )


def _float_satisfaction(preferences, satisfied: dict, students: list) -> dict:
    """Per-student float satisfaction from the honored wishes.

    The float twin of the model's integer element table: computed via
    :func:`~.satisfaction._normalize_and_bound` from the weighted honored sum
    and the student's best/worst possible sums, so the model's optimized
    integer table and this reported float agree by construction.

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
    worst_sum: dict[str, float] = {}
    for key, row in graag_met.iterrows():
        student, weight = key[0], row["Gewicht"]
        honored = satisfied[key]
        # Honored positive wish: +weight. Violated negative wish: its (negative)
        # weight. Otherwise 0 — identical to the model's weighted sum.
        contribution = weight if (weight > 0) == honored else 0.0
        honored_sum[student] = honored_sum.get(student, 0.0) + contribution
        best_sum[student] = best_sum.get(student, 0.0) + max(weight, 0.0)
        worst_sum[student] = worst_sum.get(student, 0.0) + min(weight, 0.0)

    result = {}
    for student in students:
        if student not in honored_sum:
            result[student] = 1.0
            continue
        result[student] = _normalize_and_bound(
            honored_sum[student], best_sum[student], worst_sum[student]
        )
    return result
