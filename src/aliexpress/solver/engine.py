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

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .. import errors
from ..data import preferences_data
from . import feasibility, modelbuilder, strategies
from ._balance import BalanceMaxima
from ._balance_families import SLACK_WEIGHTS, uncapped_slack_bound
from .progress import ProgressListener
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


def _floor_infeasibility_error(
    *,
    maxima: BalanceMaxima | None,
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
) -> errors.FeasibilityError:
    """The right ``FeasibilityError`` for a floor stage proven infeasible.

    With one or more families capped, the infeasibility can stem from the caps
    themselves, so :func:`.feasibility.diagnose` (which assumes balance is fully
    soft) would misattribute it to the preferences. Report an honest generic
    error in that case; the precise, actionable tip follows in a later slice.
    Without caps, the preferences are the only possible cause, so ``diagnose``
    names the family that must give.
    """
    if maxima is not None and maxima.constrains_anything():
        return errors.FeasibilityError(
            "balance_caps_infeasible",
            technical_message=(
                "Balance caps (or hard preferences) admit no valid assignment"
            ),
        )
    return errors.FeasibilityError(
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
    )


def solve_within_minimal_relaxation(  # pylint: disable=too-many-arguments
    # Each keyword-only argument is a distinct input to the model (raw data, rules,
    # strategy choice, progress listener); grouping them would obscure the entry
    # point's public interface rather than simplify it — matching the style of the
    # sibling solve_with_fixed_balance above.
    *,
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
    optimize: str = "lexmaxmin",
    listener: ProgressListener | None = None,
    maxima: BalanceMaxima | None = None,
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
    listener : ProgressListener | None
        Notified of the three UI-facing stages (``"floor"``, ``"balance"``,
        ``"satisfaction"``) as they start and finish, an interim result after each
        solved stage (the floor and balance assignments, then one per completed
        lexmaxmin level), each completed plateau, and the tie-break starting during
        ``"satisfaction"`` (see :func:`.strategies.optimize`). ``None`` (the default)
        means no one is watching; every emit site guards on it, so callers that don't
        care about progress need not pass one and pay nothing for the payloads.
    maxima : BalanceMaxima | None
        Per-family upper bounds on the relaxation. A non-empty ``maxima`` caps
        each named family's slack, so the automatic relaxation can never loosen
        that family beyond its bound (see
        :func:`.modelbuilder.build_soft_problem`). ``None`` (the default) or an
        all-``None`` ``maxima`` leaves the balance fully relaxable, as before.

    Returns
    -------
    Solution
        The solved assignment, honored wishes and recomputed satisfaction.

    Raises
    ------
    FeasibilityError
        If the first stage below comes back ``INFEASIBLE``. When ``maxima`` caps
        at least one family, this may be caused by the caps (or by the hard
        preferences) and cannot be safely attributed to preferences alone, so
        the error is a generic ``"balance_caps_infeasible"`` (the precise,
        actionable tip follows in a later slice). Otherwise the hard preference
        constraints (minimal satisfaction, not-together, "Niet in") are mutually
        infeasible even with class balance fully soft; the error is
        ``"infeasible_preferences"`` and ``context["case"]`` names the diagnosed
        cause (see :func:`.feasibility.diagnose`).
    SolverError
        If any other stage cannot be solved to proven optimality.
    """
    problem = modelbuilder.build_soft_problem(
        preferences, students, groups_to, not_together, maxima=maxima
    )
    model = problem.model

    if listener is not None:
        listener.stage_started("floor")
    t_start = time.perf_counter()
    try:
        solver = strategies.solve_stage(
            model,
            "non-positive satisfaction",
            minimize=sum(problem.nonpositive.values()),
        )
    except errors.StageInfeasible as exc:
        raise _floor_infeasibility_error(
            maxima=maxima,
            preferences=preferences,
            students=students,
            groups_to=groups_to,
            not_together=not_together,
        ) from exc
    if listener is not None:
        listener.stage_finished("floor", time.perf_counter() - t_start)
        # Every solved stage yields a complete valid assignment; report it as an
        # interim result. The floor stage's is not yet balance- or satisfaction-
        # optimized, but it is the earliest candidate to show while the rest runs.
        listener.interim_result(*problem.read_solution(solver))
    # Pin the minimal non-positive count as an upper bound for the later stages.
    model.Add(sum(problem.nonpositive.values()) <= round(solver.ObjectiveValue()))

    max_slack = model.NewIntVar(
        0, uncapped_slack_bound(students, groups_to), "max_slack"
    )
    model.AddMaxEquality(max_slack, list(problem.slacks.values()))
    weighted = (
        sum(SLACK_WEIGHTS[name] * slack for name, slack in problem.slacks.items())
        + MAX_SLACK_WEIGHT * max_slack
    )
    if listener is not None:
        listener.stage_started("balance")
    t_start = time.perf_counter()
    solver = strategies.solve_stage(model, "balance relaxation", minimize=weighted)
    if listener is not None:
        listener.stage_finished("balance", time.perf_counter() - t_start)
        listener.interim_result(*problem.read_solution(solver))
    budget = round(solver.ObjectiveValue())
    model.Add(weighted <= budget)

    if listener is not None:
        listener.stage_started("satisfaction")
    t_start = time.perf_counter()
    solver = strategies.optimize(problem, optimize, listener=listener)
    if listener is not None:
        listener.stage_finished("satisfaction", time.perf_counter() - t_start)
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
    assignment, satisfied = problem.read_solution(solver)
    return Solution(
        assignment=assignment,
        satisfied=satisfied,
        student_satisfaction=float_satisfaction(
            preferences, satisfied, list(problem.satisfaction)
        ),
    )


def float_satisfaction(preferences, satisfied: dict, students: list) -> dict:
    """Per-student float satisfaction from the honored wishes.

    The float twin of the model's integer element table: computed via
    :func:`~.satisfaction._normalize_and_bound` from the weighted honored sum
    and the student's best/worst possible sums, so the model's optimized
    integer table and this reported float agree by construction. Public (not
    underscore-prefixed) because :mod:`aliexpress.main`'s ``_InterimResultAdapter``
    also calls it, to complete an interim ``Solution`` from the preference-free
    ``assignment``/``satisfied`` a stage-boundary :meth:`~.progress.ProgressListener
    .interim_result` event carries.

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
