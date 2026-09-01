"""CP-SAT infeasibility diagnosis: which hard preference family must give?

In CP-SAT, feasibility is a plain SAT question: does any valid assignment
exist? Class balance is always soft in the real solve (relaxable via slacks),
so it can never be the cause of an infeasible instance. The remaining hard
sources are "Niet in" (fundamental, never relaxed), minimal satisfaction
floors ("Extra zekerheid") and not-together rules — the latter two are
relaxable, and this module attributes an infeasible instance to whichever of
them, once left soft, restores feasibility.
"""

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

    The first stage checks the same hard preferences with an uncapped soft
    balance model. ``None`` therefore means that the hard preferences are
    infeasible even without balance maxima. If that stage is feasible, only
    capped families receive an exact overflow variable; their weighted
    overflows are sorted and minimized leximin, with every proven level pinned.

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
    try:
        floor_solver = strategies.solve_stage(
            problem.model,
            "balance cap diagnosis floor",
            minimize=sum(problem.nonpositive.values()),
        )
    except errors.StageInfeasible:
        return None

    floor_count = round(floor_solver.ObjectiveValue())
    problem.model.Add(sum(problem.nonpositive.values()) <= floor_count)

    overflows, weighted_overflows, upper_bound = _add_cap_overflows(
        problem, students, groups_to, maxima, families
    )

    outcome = sorted_weighted_slacks.minimize_sorted_leximin(
        problem.model,
        weighted_overflows,
        max(SLACK_WEIGHTS[family] for family in FAMILY_NAMES) * upper_bound,
        solve_stage=strategies.solve_stage,
        label_prefix="balance cap overflow leximin",
        variable_prefix="cap_overflow_sort",
    )

    return _cap_suggestion(overflows, outcome.solver, maxima)
