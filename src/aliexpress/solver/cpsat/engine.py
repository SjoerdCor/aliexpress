"""The CP-SAT solve pipeline: plateaud lexmaxmin over student satisfaction.

Plateaud lexmaxmin raises the *lowest* satisfaction level by level: first lift
the minimum as high as it can go (that value is the plateau), then let as many
students as possible escape above it, pin that count, and repeat one level up
for the escapees only. This is what makes "everyone gets wish 1 before anyone
gets wish 2" the dominant behaviour. On the scaled satisfaction integers the
plateau logic is exact: "strictly above" is simply ``>= plateau + 1``.

Every stage re-solves the whole (grown) model from scratch to proven
optimality; CP-SAT needs no warm starts at this problem size.

The reported per-student satisfaction is *recomputed in float* from the honored
wishes — not read back as ``integer / SATISFACTION_SCALE`` — so the ×10^6
rounding can never leak into the report and the pinned integration values stay
exact.
"""

import logging
import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from ...data import preferences_data
from ...errors import SolverError
from ..satisfaction import get_satisfaction_integral
from . import model as cpsat_model

logger = logging.getLogger(__name__)

#: Stop raising plateaus once the minimum exceeds this satisfaction level:
#: beyond it every student has their dominant wishes honored and the
#: total-satisfaction tie-break settles the rest.
SATISFACTION_MAX = 0.8

#: At most 8 workers: measured on the herindelen benchmark (see memory of
#: 2026-07-02) more workers did not help; a fixed count keeps runs reproducible.
NUM_WORKERS = 8


@dataclass
class CpSatSolution:
    """Solved outcome, in plain Python values (no solver objects)."""

    assignment: dict  # student -> group
    satisfied: dict  # (student, Nr) -> bool (wish honored)
    student_satisfaction: dict  # student -> float, recomputed from honored wishes


def solve_with_fixed_balance(
    *, preferences, students, groups_to, not_together, groupbalance
) -> CpSatSolution:
    """Solve the distribution with hard balance limits (the manual path).

    Builds the model via :func:`.model.build_problem`, runs plateaud lexmaxmin
    plus the total-satisfaction tie-break, and returns the solved values.

    Raises
    ------
    SolverError
        If any stage cannot be solved to proven optimality.
    """
    problem = cpsat_model.build_problem(
        preferences, students, groups_to, not_together, groupbalance
    )
    _lexmaxmin(problem)
    solver = _solve_stage(
        problem.model, "tie-break", maximize=sum(problem.satisfaction.values())
    )
    return _extract(problem, solver, preferences)


def _lexmaxmin(problem) -> None:
    """Raise the minimal satisfaction level by level, pinning each plateau.

    Per level: (1) maximize the minimal satisfaction over the students above the
    previous plateau, (2) maximize how many students escape the new plateau, and
    pin that count. Stops at :data:`SATISFACTION_MAX` or when nobody escapes.
    Integer satisfaction makes both steps exact.
    """
    model = problem.model
    scale = cpsat_model.SATISFACTION_SCALE
    students = list(problem.satisfaction)
    above_plateau = {}  # students that escaped the previous plateau (empty at level 0)
    plateau = None
    level = 0
    while True:
        t_start = time.perf_counter()
        minimum = model.NewIntVar(-10 * scale, 2 * scale, f"minimum_{level}")
        if level == 0:
            for student in students:
                model.Add(minimum <= problem.satisfaction[student])
        else:
            model.Add(minimum >= plateau + 1)
            for student in students:
                model.Add(minimum <= problem.satisfaction[student]).OnlyEnforceIf(
                    above_plateau[student]
                )
        solver = _solve_stage(model, f"level {level} minimum", maximize=minimum)
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
        solver = _solve_stage(
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
        if count == 0:
            return
        model.Add(sum(above_plateau.values()) == count)
        level += 1


def _solve_stage(model, label, *, maximize):
    """Maximize the given expression to proven optimality.

    Raises :exc:`SolverError` when the stage does not reach proven optimality;
    a non-optimal stage would silently corrupt every later stage.
    """
    model.Maximize(maximize)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = NUM_WORKERS
    solver.parameters.random_seed = 1
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise SolverError(
            f"CP-SAT stage {label!r} ended with status "
            f"{solver.StatusName(status)!r}, not proven optimal"
        )
    return solver


def _extract(problem, solver, preferences) -> CpSatSolution:
    """Read the solved values; satisfaction is recomputed in float per student."""
    assignment = {
        student: group
        for (student, group), var in problem.in_group.items()
        if solver.BooleanValue(var)
    }
    satisfied = {
        key: solver.BooleanValue(literal) for key, literal in problem.satisfied.items()
    }
    return CpSatSolution(
        assignment=assignment,
        satisfied=satisfied,
        student_satisfaction=_float_satisfaction(
            preferences, satisfied, list(problem.satisfaction)
        ),
    )


def _float_satisfaction(preferences, satisfied, students) -> dict:
    """Per-student float satisfaction from the honored wishes.

    The float twin of the model's integer element table: F(weighted honored
    sum), normalized by F(best case) when the student has positive wishes, or
    added to the baseline 1 when not.
    """
    graag_met = preferences_data.get_graag_met(preferences)
    honored_sum: dict[str, float] = {}
    best_sum: dict[str, float] = {}
    for key, row in graag_met.iterrows():
        student, weight = key[0], row["Gewicht"]
        honored = satisfied[key]
        # Honored positive wish: +weight. Violated negative wish: its (negative)
        # weight. Otherwise 0 — identical to the model's weighted sum.
        contribution = weight if (weight > 0) == honored else 0.0
        honored_sum[student] = honored_sum.get(student, 0.0) + contribution
        best_sum[student] = best_sum.get(student, 0.0) + max(weight, 0.0)

    result = {}
    for student in students:
        if student not in honored_sum:
            result[student] = 1.0
            continue
        raw = get_satisfaction_integral(0, honored_sum[student])
        if best_sum[student] > 0:
            result[student] = raw / get_satisfaction_integral(0, best_sum[student])
        else:
            result[student] = 1.0 + raw
    return result
