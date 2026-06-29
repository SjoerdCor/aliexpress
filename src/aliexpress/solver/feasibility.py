"""Feasibility reasoning: relaxation budget, feasibility checks, and infeasibility diagnosis.

Functions in this module reason *about* the LP model rather than building the final
distribution.  Each one constructs a disposable :class:`pulp.LpProblem`, asks a single
question (Is it feasible? What is the minimum relaxation budget?), and returns a plain
Python value.  The :class:`~aliexpress.problemsolver.ProblemSolver` substrate is passed
in; the shared decision variables (``in_group``, ``studentsatisfaction``) are reused so
the analyses see the same model as the real solve.

See ADR-0009 for the architectural rationale (substraat-object + analyses-als-functies).
"""

import logging
from collections import defaultdict

import pulp

from .. import errors, preferences_data
from ._balance import STRICTEST_BALANCE, GroupBalance, get_solver

# Per balance slack: the GroupBalance field it relaxes and its relaxation weight.
# Whole-group ("_total") limits are cheaper to relax than per-year ones, because
# per-year balance matters more for the new cohort.
RELAXATION_WEIGHTS = {
    "SLACK_diff_n_students_year": ("max_diff_n_students_year", 1),
    "SLACK_diff_n_students_total": ("max_diff_n_students_total", 0.49),
    "SLACK_max_clique": ("max_clique", 1),
    "SLACK_max_clique_sex": ("max_clique_sex", 1),
    "SLACK_balanced_boys_girls_year": ("max_imbalance_boys_girls_year", 1),
    "SLACK_balanced_boys_girls_total": ("max_imbalance_boys_girls_total", 0.49),
}


def weighted_relaxation(prob) -> pulp.LpAffineExpression:
    """Weighted balance-relaxation expression for ``prob``.

    Returns the weighted slack sum plus the single largest slack so the relaxation
    stays spread across limits rather than piled onto one.
    """
    slacks = {v.name: v for v in prob.variables() if v.name in RELAXATION_WEIGHTS}
    max_slack = pulp.LpVariable("MAX_RELAXATION", lowBound=0)
    for slack in slacks.values():
        prob += max_slack >= slack
    weighted_sum = pulp.lpSum(
        RELAXATION_WEIGHTS[name][1] * slack for name, slack in slacks.items()
    )
    return weighted_sum + max_slack


def require_one_positive_wish(solver, prob, satisfied) -> list:
    """Softly require each student to fulfil at least one positive wish.

    Returns a per-student slack variable list, positive only for students who cannot
    structurally reach any wish (e.g. only negative preferences).  Penalizing these
    slacks keeps the requirement effective wherever achievable without making the problem
    infeasible where it is not.
    """
    positive_per_student = defaultdict(list)
    graag_met = preferences_data.get_graag_met(solver.preferences)
    for key, row in graag_met.iterrows():
        if row["Gewicht"] > 0:
            positive_per_student[key[0]].append(satisfied[key])

    wish_slacks = []
    for wishes in positive_per_student.values():
        slack = pulp.LpVariable(f"WISH_SLACK_{len(wish_slacks)}", lowBound=0)
        prob += pulp.lpSum(wishes) >= 1 - slack
        wish_slacks.append(slack)
    return wish_slacks


def minimal_relaxation_budget(
    solver, groupbalance: GroupBalance = STRICTEST_BALANCE
) -> float:
    """Return ``R*``: the smallest weighted class-balance relaxation under which every
    student can still fulfil at least one positive wish.

    Parameters
    ----------
    solver : :class:`~aliexpress.problemsolver.ProblemSolver`
        The substrate object holding the shared LP variables (``in_group``,
        ``studentsatisfaction``).  Not the pulp solver from :func:`get_solver`.
    groupbalance : :class:`~aliexpress.solver._balance.GroupBalance`
        The base balance limits to build the disposable LP around.  Defaults to
        :data:`STRICTEST_BALANCE` (all limits = 1), the tightest acceptable base for
        the normal automatic path.  A looser base allows more balance room upfront and
        produces a smaller ``R*``.  Sets ``solver.groupbalance`` as a side effect;
        ``solve_within_minimal_relaxation`` relies on this for the subsequent main solve.

    The limits are made soft, so the unmet wish slack is penalized far heavier than any
    balance relaxation: the budget is spent first on letting everyone reach a wish and
    only then, at the minimum, on extra balance room.

    This is a pure query: ``solver.groupbalance`` is temporarily set to ``groupbalance``
    for the duration of the LP build and restored afterwards, so the caller's solver
    state is unchanged.

    Raises :exc:`~aliexpress.errors.FeasibilityError` with code
    ``"infeasible_preferences"`` when the hard preference constraints are mutually
    infeasible even with balance fully relaxed.
    """
    original_groupbalance = solver.groupbalance
    solver.groupbalance = groupbalance
    try:
        prob = pulp.LpProblem("MinimalRelaxation", pulp.LpMinimize)
        solver.add_constraints(prob, make_soft=True)
        satisfied = solver.add_variables_which_preferences_satisfied(prob=prob)
        solver.calculate_student_satisfaction(satisfied, prob=prob)
        wish_slacks = require_one_positive_wish(solver, prob, satisfied)
        relaxation = weighted_relaxation(prob)
        prob.setObjective(relaxation + 1000 * pulp.lpSum(wish_slacks))
        status = prob.solve(get_solver())
        if pulp.LpStatus[status] == "Infeasible":
            # The hard preference constraints (Extra zekerheid / Niet-samen) contradict
            # each other; main.py fills in which choices clash before surfacing this.
            raise errors.FeasibilityError(
                "infeasible_preferences",
                technical_message="Hard preference constraints are mutually infeasible",
            )
        if pulp.LpStatus[status] != "Optimal":
            raise ValueError("Could not determine the minimal class-balance relaxation")
        return relaxation.value()
    finally:
        solver.groupbalance = original_groupbalance


def check_balance_feasibility(solver) -> "pulp.LpProblem":
    """Return the solved balance-feasibility LP for ``solver``'s current groupbalance.

    ``solver`` is a :class:`~aliexpress.problemsolver.ProblemSolver` instance.

    Builds a disposable LP with all constraints made soft (balance slacks enabled),
    minimises the total slack, solves it, and returns the LP so the caller can inspect
    individual variable values.  An objective value of 0 means the fixed groupbalance is
    exactly feasible; a positive value means at least one balance limit must give.

    Used by ``main._check_feasibility`` on the manual path (``groupbalance`` supplied by
    the caller); on the automatic path ``solve_within_minimal_relaxation`` handles balance
    adaptation directly.
    """
    feas_prob = pulp.LpProblem("MinimumRelaxationFeasibility", pulp.LpMinimize)
    solver.add_constraints(feas_prob, make_soft=True)
    slack_vars = [v for v in feas_prob.variables() if "SLACK" in v.name]
    feas_prob.setObjective(pulp.lpSum(slack_vars))
    feas_prob.solve(get_solver())

    _log = logging.getLogger(__name__)
    if feas_prob.objective.value() == 0:
        _log.info("Problem feasible. Continue")
    else:
        msg = "Problem infeasible. Consider changing variables to make it possible:\n"
        for v in slack_vars:
            if v.value() > 0:
                msg += f'{v.name.lstrip("SLACK_")}: +{round(v.value())}\n'
        _log.error(msg)
    return feas_prob


def feasible_when_relaxed(
    solver, *, min_satisfaction_soft: bool, not_together_soft: bool
) -> bool:
    """Whether a feasible assignment exists when the chosen preference families are soft.

    ``solver`` is a :class:`~aliexpress.problemsolver.ProblemSolver` instance (not the
    pulp solver from :func:`get_solver`).  Class balance is kept soft throughout (as in
    the real solve), so infeasibility can only stem from the preference families left
    hard.  Used by :func:`diagnose` to attribute infeasibility to a family.
    """
    prob = pulp.LpProblem("DiagnoseFeasibility", pulp.LpMinimize)
    solver.add_fundamental_constraints(prob)
    solver.add_class_balance_constraints(prob, make_soft=True)
    satisfied = solver.add_variables_which_preferences_satisfied(prob=prob)
    solver.calculate_student_satisfaction(satisfied, prob=prob)
    slacks = solver.constraint_minimal_satisfaction(
        prob, make_soft=min_satisfaction_soft
    ) + solver.constraint_not_together(prob, make_soft=not_together_soft)
    prob.setObjective(pulp.lpSum(slacks))
    status = prob.solve(get_solver())
    return pulp.LpStatus[status] == "Optimal"


def diagnose(solver) -> str:
    """Return which preference family must give for the distribution to become feasible.

    ``solver`` is a :class:`~aliexpress.problemsolver.ProblemSolver` instance.

    One of ``"min_satisfaction"`` / ``"not_together"`` (relaxing that family alone
    suffices), ``"either"`` (each alone suffices), ``"both"`` (only relaxing both helps)
    or ``"fundamental"`` (relaxing both still fails, so the cause lies elsewhere, e.g. a
    ``"Niet in"``-exclusion).
    """
    min_sat_helps = feasible_when_relaxed(
        solver, min_satisfaction_soft=True, not_together_soft=False
    )
    not_together_helps = feasible_when_relaxed(
        solver, min_satisfaction_soft=False, not_together_soft=True
    )
    if min_sat_helps and not_together_helps:
        return "either"
    if min_sat_helps:
        return "min_satisfaction"
    if not_together_helps:
        return "not_together"
    if feasible_when_relaxed(
        solver, min_satisfaction_soft=True, not_together_soft=True
    ):
        return "both"
    return "fundamental"
