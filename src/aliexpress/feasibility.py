"""Feasibility reasoning: relaxation budget, feasibility checks, and infeasibility diagnosis.

Functions in this module reason *about* the LP model rather than building the final
distribution.  Each one constructs a disposable :class:`pulp.LpProblem`, asks a single
question (Is it feasible? What is the minimum relaxation budget?), and returns a plain
Python value.  The :class:`~aliexpress.problemsolver.ProblemSolver` substrate is passed
in; the shared decision variables (``in_group``, ``studentsatisfaction``) are reused so
the analyses see the same model as the real solve.

See ADR-0009 for the architectural rationale (substraat-object + analyses-als-functies).
"""

import pulp

from .problemsolver import get_solver


def feasible_when_relaxed(
    solver, *, min_satisfaction_soft: bool, not_together_soft: bool
) -> bool:
    """Whether a feasible assignment exists when the chosen families are made soft.

    Class balance is kept soft throughout (as in the real solve), so infeasibility can
    only stem from the preference families left hard.  Used by :func:`diagnose` to
    attribute infeasibility to a family.
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
    """Return which preference family must give for ``solver`` to become feasible.

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
