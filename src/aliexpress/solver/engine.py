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
from . import feasibility, modelbuilder
from . import sorted_weighted_slacks as sorted_weighted_slacks_module
from . import strategies
from ._balance import UNCAPPED, BalanceMaxima
from ._balance_families import SLACK_WEIGHTS, slack_upper_bounds, uncapped_slack_bound
from .progress import ProgressListener
from .satisfaction import _normalize_and_bound


@dataclass
class Solution:
    """Solved outcome, in plain Python values (no solver objects)."""

    assignment: dict  # student -> group
    satisfied: dict  # (student, Nr) -> bool (wish honored)
    student_satisfaction: dict  # student -> float, recomputed from honored wishes


@dataclass
class _BalanceLeximinOutcome:
    """Proven sorted weighted slacks plus the solver that produced the last value."""

    solver: cp_model.CpSolver
    sorted_weighted_slacks: tuple[int, ...]


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
    maxima: BalanceMaxima,
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
) -> errors.FeasibilityError:
    """The right ``FeasibilityError`` for a floor stage proven infeasible.

    With one or more families capped, the infeasibility can stem from the caps
    themselves, so first compare the capped model with an otherwise identical
    uncapped model. A joint weighted-leximin overflow suggestion proves that
    the caps are the cause. If the uncapped model is infeasible too,
    :func:`.feasibility.diagnose` diagnoses the hard preferences instead.
    Without caps, the preferences are the only possible cause, so ``diagnose``
    names the family that must give.
    """
    if maxima.constrains_anything():
        suggestion = feasibility.diagnose_balance_caps(
            preferences=preferences,
            students=students,
            groups_to=groups_to,
            not_together=not_together,
            maxima=maxima,
        )
        if suggestion is not None:
            return errors.FeasibilityError(
                "balance_caps_too_tight",
                context={"suggestion": suggestion},
                technical_message="Configured balance caps are too tight",
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
    maxima: BalanceMaxima = UNCAPPED,
) -> Solution:
    """Solve the distribution with the class balance relaxed only as far as needed.

    Builds the model via :func:`.modelbuilder.build_soft_problem` and fixes the class
    balance in two lexicographic stages before the main solve:

    1. Minimize the number of students left at or below zero satisfaction
       (normally 0), then pin that count as an upper bound. A student cannot
       reach strictly positive satisfaction if the balance that would give
       them one is forbidden, so this stage finds how much relaxation is
       unavoidable.
    2. With that count pinned, minimize the six weighted balance slacks in
       descending order, leximin, and pin the resulting sorted weighted slacks
       too
       (see :func:`_pin_balance_leximin`). Whole-group limits weigh less than
       per-year ones (:data:`~._balance_families.SLACK_WEIGHTS`); leximin
       spreads the relaxation across limits rather than piling it onto one.

    The chosen strategy then runs on a freshly built model carrying only the
    proven floor and exact sorted weighted slacks. This preserves every distribution
    admitted by those decisions while dropping the balance-optimization-only
    sorting variables and constraints that would otherwise burden every
    satisfaction sub-stage.

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
    maxima : BalanceMaxima
        Per-family upper bounds on the relaxation. A non-empty ``maxima`` caps
        each named family's slack, so the automatic relaxation can never loosen
        that family beyond its bound (see
        :func:`.modelbuilder.build_soft_problem`). An empty (all-unlimited)
        ``BalanceMaxima`` (the default) leaves the balance fully relaxable.

    Returns
    -------
    Solution
        The solved assignment, honored wishes and recomputed satisfaction.

    Raises
    ------
    FeasibilityError
        If the first stage below comes back ``INFEASIBLE``. When ``maxima`` caps
        at least one family, a silent uncapped comparison either returns one
        joint weighted-leximin relaxation in ``context["suggestion"]`` with
        code ``"balance_caps_too_tight"``, or proves the hard preferences
        infeasible and returns ``"infeasible_preferences"``. Without caps, the
        latter existing preference diagnosis is used directly. The diagnosis
        emits no progress events.
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
    floor_count = round(solver.ObjectiveValue())
    model.Add(sum(problem.nonpositive.values()) <= floor_count)

    if listener is not None:
        listener.stage_started("balance")
    t_start = time.perf_counter()
    balance_outcome = _pin_balance_leximin(problem, students, groups_to)
    if listener is not None:
        listener.stage_finished("balance", time.perf_counter() - t_start)
        listener.interim_result(*problem.read_solution(balance_outcome.solver))

    # CP-SAT starts every stage from a fresh search; keeping the sorting network
    # and all of its balance-optimization-only variables and constraints in the
    # satisfaction phase only makes each plateau re-process that machinery.
    # Rebuild the base problem and carry
    # over exactly the two proven decisions: the floor count and the sorted
    # weighted slacks. The table preserves every valid mapping of sorted
    # positions to balance families.
    problem = _build_sorted_weighted_slacks_pinned_problem(
        preferences=preferences,
        students=students,
        groups_to=groups_to,
        not_together=not_together,
        maxima=maxima,
        floor_count=floor_count,
        sorted_weighted_slacks=balance_outcome.sorted_weighted_slacks,
    )

    if listener is not None:
        listener.stage_started("satisfaction")
    t_start = time.perf_counter()
    solver = strategies.optimize(problem, optimize, listener=listener)
    if listener is not None:
        listener.stage_finished("satisfaction", time.perf_counter() - t_start)
    return _extract(problem, solver, preferences)


def _build_sorted_weighted_slacks_pinned_problem(  # pylint: disable=too-many-arguments
    # Rebuilding requires the same five independent model inputs as the public
    # solve entry point, plus the two proven decisions being transferred.
    # Wrapping those once-used values in another object would hide this boundary
    # without reducing the data the operation actually needs.
    *,
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
    maxima: BalanceMaxima,
    floor_count: int,
    sorted_weighted_slacks: tuple[int, ...],
) -> modelbuilder.SoftProblem:
    """Build a clean satisfaction model with the proven pre-stages pinned."""
    problem = modelbuilder.build_soft_problem(
        preferences, students, groups_to, not_together, maxima=maxima
    )
    problem.model.Add(sum(problem.nonpositive.values()) <= floor_count)
    sorted_weighted_slacks_module.pin_exact_sorted_weighted_slacks(
        problem.model,
        problem.slacks,
        sorted_weighted_slacks,
        slack_upper_bounds(students, groups_to, maxima),
    )
    return problem


def _pin_balance_leximin(
    problem, students: dict, groups_to: dict
) -> _BalanceLeximinOutcome:
    """Minimize and pin the sorted weighted slacks, using leximin.

    Rather than one weighted sum (which lets CP-SAT trade slack fractionally
    between families without a provable optimum once caps bind — see
    ADR-0018), this minimizes the ``weight * slack`` values sorted from largest
    to smallest, one entry at a time. A fixed compare-swap network materializes
    that ordering once; stage ``k`` then pins its ``(k+1)``-th output to the
    minimal value. Which family occupies which position remains free: only the
    sorted weighted slacks are pinned, not the assignment of slack to family.
    Stops as soon as a pinned value is 0, since the remaining entries must be 0
    too.
    Each sub-stage runs via :func:`.strategies.solve_stage`, so it is proven
    optimal or raises, same guarantee as the single stage it replaces.

    Parameters
    ----------
    problem : modelbuilder.SoftProblem
        The built soft problem, for its ``model`` and per-family ``slacks``.
    students : dict
        Per-student info, forwarded to :func:`uncapped_slack_bound` for the
        domain of every ``M_k`` variable.
    groups_to : dict
        Target groups with current occupancy, forwarded the same way.

    Returns
    -------
    _BalanceLeximinOutcome
        The full proven sorted weighted slacks and the solver holding the last
        sub-stage's optimal solution.
    """
    model = problem.model
    weighted = {
        name: SLACK_WEIGHTS[name] * slack for name, slack in problem.slacks.items()
    }
    m_upper = max(SLACK_WEIGHTS.values()) * uncapped_slack_bound(students, groups_to)
    outcome = sorted_weighted_slacks_module.minimize_sorted_leximin(
        model,
        list(weighted.values()),
        m_upper,
        solve_stage=strategies.solve_stage,
        label_prefix="balance leximin",
        variable_prefix="balance_sort",
    )
    return _BalanceLeximinOutcome(outcome.solver, outcome.values)


def _sorting_network_descending(
    model: cp_model.CpModel, values: list, upper_bound: int
) -> list[cp_model.IntVar]:
    """Materialize ``values`` in descending order with fixed compare-swaps.

    This is an insertion sorting network: each new value passes through the
    already-sorted prefix, and every comparator emits an exact ``max`` followed
    by an exact ``min``. For the six balance families it needs only fifteen
    comparators. Unlike per-rank ``exceed`` booleans, the network establishes
    every order statistic once and propagates bounds between adjacent sorted
    weighted-slack positions throughout all later solve stages.
    """
    return sorted_weighted_slacks_module.sorting_network_descending(
        model, values, upper_bound
    )


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
