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

from . import model as cpsat_model
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

    Builds :func:`.model.build_feasibility_problem` with the two relaxable
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
    problem = cpsat_model.build_feasibility_problem(
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
