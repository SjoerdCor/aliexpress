"""Attribute an infeasible automatic-path solve to a preference family (see ADR-0008).

When ``ProblemSolver._minimal_relaxation_budget`` is infeasible, the cause is a clash of
the hard preference families: minimal satisfaction (UI: "Extra zekerheid") and the
not-together rules (UI: "Niet-samen"). This module runs a family-level leave-one-out via
``ProblemSolver.feasible_when_relaxed`` and reports which family must give.

Family level is deliberate: a minimal relaxation is often degenerate, so naming a single
arbitrary student/rule would suggest a unique culprit that does not exist. Which *family*
is necessary/sufficient is well-defined.
"""


def diagnose(solver) -> str:
    """Return which preference family must give for ``solver`` to become feasible.

    One of ``"min_satisfaction"`` / ``"not_together"`` (relaxing that family alone
    suffices), ``"either"`` (each alone suffices), ``"both"`` (only relaxing both helps) or
    ``"fundamental"`` (relaxing both still fails, so the cause lies elsewhere, e.g. a
    "Niet in"-exclusion).
    """
    min_sat_helps = solver.feasible_when_relaxed(
        min_satisfaction_soft=True, not_together_soft=False
    )
    not_together_helps = solver.feasible_when_relaxed(
        min_satisfaction_soft=False, not_together_soft=True
    )
    if min_sat_helps and not_together_helps:
        return "either"
    if min_sat_helps:
        return "min_satisfaction"
    if not_together_helps:
        return "not_together"
    if solver.feasible_when_relaxed(min_satisfaction_soft=True, not_together_soft=True):
        return "both"
    return "fundamental"
