"""CP-SAT infeasibility diagnosis: which hard preference family must give?

In CP-SAT, feasibility is a plain SAT question: does any valid assignment
exist? Class balance is always soft in the real solve (relaxable via slacks),
so it can never be the cause of an infeasible instance. The remaining hard
sources are "Niet in" (fundamental, never relaxed), minimal satisfaction
floors ("Extra zekerheid") and not-together rules — the latter two are
relaxable, and this module attributes an infeasible instance to whichever of
them, once left soft, restores feasibility.
"""

import time

from ortools.sat.python import cp_model

from .. import errors
from . import modelbuilder, sorted_weighted_slacks, strategies
from ._balance import UNCAPPED, BalanceMaxima
from ._balance_families import (
    FAMILY_NAMES,
    SLACK_WEIGHTS,
    STRICTEST_LIMIT,
    capped_families,
    maximum_for_family,
    uncapped_slack_bound,
)
from .conflicts import Conflict
from .strategies import NUM_WORKERS


def feasible_when_relaxed(  # pylint: disable=too-many-arguments
    # Each keyword-only argument is a distinct input to the model (raw data,
    # rules, which relaxable families to leave soft); grouping them would
    # obscure the function's public interface rather than simplify it.
    *,
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
    min_satisfaction_soft: bool,
    not_together_soft: bool,
) -> bool:
    """Whether a valid assignment exists when the chosen families are left soft.

    Builds :func:`.modelbuilder.build_feasibility_problem` with the two relaxable
    families hard except where the caller marks them soft, and checks whether
    any assignment satisfies it. Class balance is always soft and "Niet in" is
    always hard, matching the real solve.

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
    min_satisfaction_soft : bool
        Whether the minimal-satisfaction floors are left unenforced.
    not_together_soft : bool
        Whether the not-together rules are left unenforced.

    Returns
    -------
    bool
        Whether a valid assignment exists under this relaxation choice.
    """
    problem = modelbuilder.build_feasibility_problem(
        preferences,
        students,
        groups_to,
        not_together,
        min_satisfaction_hard=not min_satisfaction_soft,
        not_together_hard=not not_together_soft,
    )
    solver = cp_model.CpSolver()
    # A feasibility verdict ("does any assignment exist?") is a property of the
    # model itself, not of the search that finds it — so it cannot depend on
    # the worker count. NUM_WORKERS and a fixed seed are set anyway, purely to
    # keep the run time of this check reproducible across machines.
    solver.parameters.num_workers = NUM_WORKERS
    solver.parameters.random_seed = 1
    status = solver.Solve(problem)
    return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def diagnose_conflict(  # pylint: disable=too-many-arguments
    *,
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
    deadline_seconds: float = 10.0,
) -> Conflict | None:
    """Find one fully checked, subset-minimal conflict, or return ``None``.

    The initial solve asks CP-SAT for a sufficient assumption core.  Only that core is
    tested further: each candidate is removed in stable input order only when the
    remaining assumptions are still infeasible.  Every solve must finish with a
    decisive status before a conflict can be returned; timeout, ``UNKNOWN``, an invalid
    model or an invalid/empty solver core therefore produces the safe fallback signal
    ``None``.
    """
    if deadline_seconds <= 0:
        return None

    started = time.perf_counter()
    try:
        problem = modelbuilder.build_diagnostic_problem(
            preferences, students, groups_to, not_together
        )
    except Exception:  # pylint: disable=broad-exception-caught
        # A diagnosis is optional; a builder failure must preserve the normal fallback.
        return None

    def solve(active_indices):
        return _solve_diagnostic(
            problem,
            active_indices,
            _remaining_time(started, deadline_seconds),
        )

    core = _extract_sufficient_core(problem, solve)
    if core is None:
        return None
    core = _shrink_core(core, solve)
    if core is None:
        return None
    return Conflict(tuple(problem.condition_by_index[index] for index in core))


def _extract_sufficient_core(problem, solve):
    """Return CP-SAT's sufficient core in stable condition order, if proven."""
    status, solver = solve(list(problem.literal_by_index))
    if status != cp_model.INFEASIBLE:
        return None

    try:
        solver_core = list(solver.SufficientAssumptionsForInfeasibility())
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    if not solver_core or any(
        index not in problem.condition_by_index for index in solver_core
    ):
        return None

    # CP-SAT may return a valid core in a solver-dependent order.  The model builder's
    # insertion order is the stable condition order used for all shrink checks and output.
    order = {
        index: position for position, index in enumerate(problem.condition_by_index)
    }
    return sorted(set(solver_core), key=order.__getitem__)


def _shrink_core(core, solve):
    """Remove redundant conditions while every check returns a decisive status."""
    # Greedily remove each candidate.  A condition is necessary when removing it makes
    # the remaining assumptions feasible; otherwise it is not part of this irreducible
    # conflict.  ``tuple(core)`` is a snapshot because ``core`` shrinks in the loop.
    for index in tuple(core):
        candidate = [
            candidate_index for candidate_index in core if candidate_index != index
        ]
        status, _ = solve(candidate)
        if status == cp_model.INFEASIBLE:
            core.remove(index)
        elif status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        else:
            return None
    return core


def _remaining_time(started: float, deadline_seconds: float) -> float:
    """Return the positive wall-clock budget left for one diagnostic solve."""
    return deadline_seconds - (time.perf_counter() - started)


def _solve_diagnostic(problem, active_indices, remaining_seconds):
    """Solve a diagnostic model with exactly ``active_indices`` assumptions."""
    if remaining_seconds <= 0:
        return cp_model.UNKNOWN, None
    problem.model.ClearAssumptions()
    problem.model.AddAssumptions(
        [problem.literal_by_index[index] for index in active_indices]
    )
    solver = cp_model.CpSolver()
    # Diagnosis is deliberately reproducible and independent from normal solve settings.
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 1
    solver.parameters.max_time_in_seconds = remaining_seconds
    try:
        return solver.Solve(problem.model), solver
    except Exception:  # pylint: disable=broad-exception-caught
        return cp_model.MODEL_INVALID, solver


def diagnose(
    *, preferences, students: dict, groups_to: dict, not_together: list
) -> str:
    """Attribute an infeasible instance to the hard family that must give.

    Assumes the instance is already known to be infeasible with both
    relaxable families hard; only called from that error path.

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

    Returns
    -------
    str
        ``"min_satisfaction"`` or ``"not_together"`` when relaxing that family
        alone suffices, ``"either"`` when each alone suffices, ``"both"`` when
        only relaxing both together suffices, or ``"fundamental"`` when even
        that does not help (the cause lies elsewhere, e.g. a "Niet in").
    """
    common = {
        "preferences": preferences,
        "students": students,
        "groups_to": groups_to,
        "not_together": not_together,
    }
    # Leave-one-out first: each of the two relaxable families tried alone.
    single_case = {
        (True, False): "min_satisfaction",
        (False, True): "not_together",
        (True, True): "either",
    }
    helps = (
        feasible_when_relaxed(
            **common, min_satisfaction_soft=True, not_together_soft=False
        ),
        feasible_when_relaxed(
            **common, min_satisfaction_soft=False, not_together_soft=True
        ),
    )
    if helps in single_case:
        return single_case[helps]
    # Neither alone helped; only a case for "both" remains before "fundamental".
    both_help = feasible_when_relaxed(
        **common, min_satisfaction_soft=True, not_together_soft=True
    )
    return "both" if both_help else "fundamental"


def _add_cap_overflows(problem, students, groups_to, maxima, families):
    """Add exact overflow variables for the capped families only."""
    upper_bound = uncapped_slack_bound(students, groups_to)
    overflows = {}
    weighted_overflows = []
    for family in families:
        cap = maximum_for_family(family, maxima)
        cap_slack = cap - STRICTEST_LIMIT
        overflow = problem.model.NewIntVar(0, upper_bound, f"cap_overflow_{family}")
        problem.model.AddMaxEquality(overflow, [0, problem.slacks[family] - cap_slack])
        overflows[family] = overflow
        weighted_overflows.append(SLACK_WEIGHTS[family] * overflow)
    return overflows, weighted_overflows, upper_bound


def _cap_suggestion(overflows, solver, maxima):
    """Read positive overflow values as current/suggested cap pairs."""
    suggestion = {}
    for family, overflow in overflows.items():
        amount = solver.Value(overflow)
        if amount > 0:
            current = maximum_for_family(family, maxima)
            suggestion[family] = {
                "current": current,
                "suggested": current + amount,
            }
    return suggestion


def diagnose_balance_caps(  # pylint: disable=too-many-arguments
    *,
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
    maxima: BalanceMaxima,
) -> dict | None:
    """Find one joint weighted-leximin loosening for infeasible balance caps.

    The uncapped soft-balance model contains the same hard constraints as the
    failed capped model. Only capped families receive an exact overflow
    variable; their weighted overflows are sorted and minimized leximin, with
    every proven level pinned. If an initial feasibility check fails, the hard
    preferences are infeasible without balance maxima too and this function
    returns ``None``.

    Deliberately do not optimize or pin the number of students without a
    positive wish here. The failed floor stage proves that the configured caps
    admit no valid assignment at all. A cap suggestion therefore has to restore
    feasibility, not preserve the better satisfaction floor of an uncapped
    model; requiring the latter can add increases that are unnecessary for a
    valid run.

    The diagnosis is deliberately not a progress phase: it has no listener and
    emits no interim result. A later unexpected infeasibility is allowed to
    propagate instead of being misclassified as a preference problem.
    """
    families = capped_families(maxima)
    if not families:
        return {}

    problem = modelbuilder.build_soft_problem(
        preferences, students, groups_to, not_together, maxima=UNCAPPED
    )
    overflows, weighted_overflows, upper_bound = _add_cap_overflows(
        problem, students, groups_to, maxima, families
    )

    try:
        strategies.solve_stage(
            problem.model, "balance cap diagnosis feasibility", minimize=0
        )
    except errors.StageInfeasible:
        return None

    outcome = sorted_weighted_slacks.minimize_sorted_leximin(
        problem.model,
        weighted_overflows,
        max(SLACK_WEIGHTS[family] for family in FAMILY_NAMES) * upper_bound,
        solve_stage=strategies.solve_stage,
        label_prefix="balance cap overflow leximin",
        variable_prefix="cap_overflow_sort",
    )

    return _cap_suggestion(overflows, outcome.solver, maxima)
