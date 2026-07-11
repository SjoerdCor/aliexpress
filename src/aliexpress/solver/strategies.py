"""From per-student satisfaction to a single CP-SAT objective.

Given the model's per-student satisfaction integers, this module answers one
question: what aggregate value does the solver optimize? Two strategies are
implemented, chosen via :func:`optimize`'s ``strategy`` argument:

- ``"total"``: maximize the sum of all satisfaction integers, directly.
- ``"lexmaxmin"``: plateaud lexicographic max-min. Raises the *lowest*
  satisfaction level by level: first lift the minimum as high as it can go
  (that value is the plateau), then let as many students as possible escape
  above it, pin that count, and repeat one level up for the escapees only.
  This is what makes "everyone gets wish 1 before anyone gets wish 2" the
  dominant behaviour. Once every plateau is pinned, the total-satisfaction sum
  still runs as a final tie-break — the same objective ``"total"`` uses on its
  own, just applied after the plateau constraints instead of them. On the
  scaled satisfaction integers the plateau logic is exact: "strictly above" is
  simply ``>= plateau + 1``.

Boundary with ``modelbuilder.py``: that module builds the constraints and the
per-student satisfaction integers (the *problem*). This module aggregates
those integers across students into a single objective (the *strategy*), and
owns the generic "solve to proven optimality" helper every stage — of either
strategy, or of the caller's own lexicographic stages — re-uses.

Every stage re-solves the whole (grown) model from scratch to proven
optimality; CP-SAT needs no warm starts at this problem size.
"""

import logging
import time

from ortools.sat.python import cp_model

from ..errors import SolverError, StageInfeasible
from . import modelbuilder
from .progress import ProgressListener

logger = logging.getLogger(__name__)

#: Stop raising plateaus once the minimum exceeds this satisfaction level:
#: beyond it every student has their dominant wishes honored and the
#: total-satisfaction tie-break settles the rest.
SATISFACTION_MAX = 0.8

#: At most 8 workers: measured on the herindelen benchmark (see memory of
#: 2026-07-02) more workers did not help; a fixed count keeps runs reproducible.
NUM_WORKERS = 8


def optimize(
    problem, strategy: str, listener: ProgressListener | None = None
) -> cp_model.CpSolver:
    """Run the chosen aggregate objective and return the final-stage solver.

    Parameters
    ----------
    problem : modelbuilder.Problem | modelbuilder.SoftProblem
        The built model; mutated in place by ``"lexmaxmin"``'s plateau stages.
    strategy : str
        The strategy to run: ``"lexmaxmin"`` or ``"total"`` (see the module
        docstring).
    listener : ProgressListener | None
        Notified of each completed lexmaxmin plateau and of the tie-break
        starting; not consulted by the ``"total"`` strategy, which has neither.
        Defaults to the no-op base class.

    Returns
    -------
    cp_model.CpSolver
        The solver holding the proven-optimal final solution.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of the two known strategies.
    """
    listener = listener or ProgressListener()
    if strategy == "lexmaxmin":
        _lexmaxmin(problem, listener)
        listener.tiebreak_started()
        return solve_stage(
            problem.model, "tie-break", maximize=sum(problem.satisfaction.values())
        )
    if strategy == "total":
        return solve_stage(
            problem.model, "total", maximize=sum(problem.satisfaction.values())
        )
    raise ValueError(f"unknown optimize strategy {strategy!r}")


def _lexmaxmin(problem, listener: ProgressListener) -> None:
    """Raise the minimal satisfaction level by level, pinning each plateau.

    Per level: (1) maximize the minimal satisfaction over the students above the
    previous plateau, (2) maximize how many students escape the new plateau, and
    pin that count. Stops at :data:`SATISFACTION_MAX` or when nobody escapes.
    Integer satisfaction makes both steps exact. ``listener.plateau_finished`` fires
    once per completed level (both stages solved), including the terminal level
    where nobody escapes — but not on the early :data:`SATISFACTION_MAX` return,
    which stops before the count stage runs.

    Parameters
    ----------
    problem : modelbuilder.Problem | modelbuilder.SoftProblem
        The built model; mutated in place with the plateau constraints.
    listener : ProgressListener
        Notified via ``plateau_finished(min_satisfaction, n_can_improve)`` after
        each completed level.
    """
    model = problem.model
    scale = modelbuilder.SATISFACTION_SCALE
    students = list(problem.satisfaction)
    # The minimum can only ever equal one student's satisfaction, so its domain
    # is exactly the union of every satisfaction variable's own domain — data-
    # driven, unlike a fixed constant: a student with only violated negative
    # wishes can score far below any small fixed lower bound.
    lower_bound = min(low for low, _ in problem.satisfaction_bounds.values())
    upper_bound = max(high for _, high in problem.satisfaction_bounds.values())
    above_plateau = {}  # students that escaped the previous plateau (empty at level 0)
    plateau = None
    level = 0
    while True:
        t_start = time.perf_counter()
        minimum = model.NewIntVar(lower_bound, upper_bound, f"minimum_{level}")
        if level == 0:
            for student in students:
                model.Add(minimum <= problem.satisfaction[student])
        else:
            model.Add(minimum >= plateau + 1)
            for student in students:
                model.Add(minimum <= problem.satisfaction[student]).OnlyEnforceIf(
                    above_plateau[student]
                )
        solver = solve_stage(model, f"level {level} minimum", maximize=minimum)
        plateau = round(solver.Value(minimum))
        if plateau > SATISFACTION_MAX * scale:
            logger.debug("lexmaxmin stopped: minimum above %s", SATISFACTION_MAX)
            return
        if level == 0:
            for student in students:
                model.Add(problem.satisfaction[student] >= plateau)
        else:
            for student in students:
                model.Add(problem.satisfaction[student] >= plateau).OnlyEnforceIf(
                    above_plateau[student]
                )

        above_plateau = {
            student: model.NewBoolVar(f"above_{level}_{student}")
            for student in students
        }
        for student in students:
            model.Add(problem.satisfaction[student] >= plateau + 1).OnlyEnforceIf(
                above_plateau[student]
            )
            model.Add(problem.satisfaction[student] <= plateau).OnlyEnforceIf(
                above_plateau[student].Not()
            )
        solver = solve_stage(
            model, f"level {level} count", maximize=sum(above_plateau.values())
        )
        count = round(solver.ObjectiveValue())
        logger.info(
            "lexmaxmin level %d: plateau=%.6f, %d above, %.2fs",
            level,
            plateau / scale,
            count,
            time.perf_counter() - t_start,
        )
        listener.plateau_finished(plateau / scale, count)
        if count == 0:
            return
        model.Add(sum(above_plateau.values()) == count)
        level += 1


def solve_stage(
    model: cp_model.CpModel,
    label: str,
    *,
    maximize: cp_model.LinearExprT | None = None,
    minimize: cp_model.LinearExprT | None = None,
) -> cp_model.CpSolver:
    """Optimize the given expression to proven optimality.

    Exactly one of ``maximize``/``minimize`` is given by the caller. Shared by
    both strategies here and by the caller's own lexicographic stages (fixing
    the minimal-relaxation balance before the strategy runs).

    A proven-infeasible stage raises :class:`StageInfeasible`, distinct from the
    ``SolverError`` any other non-optimal status raises. Most callers let both
    propagate; a caller for whom infeasibility means something concrete (the
    automatic path's first stage: the hard preferences contradict each other)
    catches :class:`StageInfeasible` specifically.

    Parameters
    ----------
    model : cp_model.CpModel
        The model to solve; its objective is set as a side effect.
    label : str
        Identifies the stage in the raised error message.
    maximize : cp_model.LinearExprT | None
        The expression to maximize, or ``None`` when ``minimize`` is given.
    minimize : cp_model.LinearExprT | None
        The expression to minimize, or ``None`` when ``maximize`` is given.

    Returns
    -------
    cp_model.CpSolver
        The solver holding the proven-optimal solution.

    Raises
    ------
    ValueError
        If not exactly one of ``maximize``/``minimize`` is given.
    StageInfeasible
        If the stage is proven ``INFEASIBLE``.
    SolverError
        If the stage reaches any other non-optimal status; a non-optimal stage
        would silently corrupt every later stage.
    """
    if (maximize is None) == (minimize is None):
        raise ValueError(
            f"CP-SAT stage {label!r}: pass exactly one of maximize/minimize"
        )
    if minimize is not None:
        model.Minimize(minimize)
    else:
        model.Maximize(maximize)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = NUM_WORKERS
    solver.parameters.random_seed = 1
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        raise StageInfeasible(label)
    if status != cp_model.OPTIMAL:
        raise SolverError(
            f"CP-SAT stage {label!r} ended with status "
            f"{solver.StatusName(status)!r}, not proven optimal"
        )
    return solver
