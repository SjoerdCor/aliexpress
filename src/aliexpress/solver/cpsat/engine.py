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
from ._balance_families import SLACK_WEIGHTS

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
    *,
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
    groupbalance,
) -> CpSatSolution:
    """Solve the distribution with hard balance limits (the manual path).

    Builds the model via :func:`.model.build_problem`, runs plateaud lexmaxmin
    plus the total-satisfaction tie-break, and returns the solved values.

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

    Returns
    -------
    CpSatSolution
        The solved assignment, honored wishes and recomputed satisfaction.

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


def solve_within_minimal_relaxation(
    *,
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
) -> CpSatSolution:
    """Solve the distribution with the class balance relaxed only as far as needed.

    Builds the model via :func:`.model.build_soft_problem` and fixes the class
    balance in two lexicographic stages before the main solve:

    1. Minimize the number of students left without any honored positive wish
       (normally 0), then pin that count as an upper bound. A student cannot
       keep a positive wish if the balance that would give it to them is
       forbidden, so this stage finds how much relaxation is unavoidable.
    2. With that count pinned, minimize the weighted balance relaxation, then
       pin the resulting minimum too. Whole-group limits weigh less than
       per-year ones (:data:`~._balance_families.SLACK_WEIGHTS`), and a
       max-slack term keeps the relaxation spread across limits rather than
       piled onto one.

    Plateaud lexmaxmin plus the total-satisfaction tie-break then run on the
    same model, now that the class balance is fixed at its minimal relaxation.

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
    CpSatSolution
        The solved assignment, honored wishes and recomputed satisfaction.

    Raises
    ------
    SolverError
        If any stage cannot be solved to proven optimality.
    """
    problem = cpsat_model.build_soft_problem(
        preferences, students, groups_to, not_together
    )
    model = problem.model

    solver = _solve_stage(model, "unmet wishes", minimize=sum(problem.unmet.values()))
    unmet_optimum = round(solver.ObjectiveValue())
    model.Add(sum(problem.unmet.values()) <= unmet_optimum)

    max_slack = model.NewIntVar(0, len(students), "max_slack")
    model.AddMaxEquality(max_slack, list(problem.slacks.values()))
    weighted = (
        sum(SLACK_WEIGHTS[name] * slack for name, slack in problem.slacks.items())
        + 100 * max_slack
    )
    solver = _solve_stage(model, "balance relaxation", minimize=weighted)
    budget = round(solver.ObjectiveValue())
    model.Add(weighted <= budget)

    _lexmaxmin(problem)
    solver = _solve_stage(
        model, "tie-break", maximize=sum(problem.satisfaction.values())
    )
    return _extract(problem, solver, preferences)


def _lexmaxmin(problem) -> None:
    """Raise the minimal satisfaction level by level, pinning each plateau.

    Per level: (1) maximize the minimal satisfaction over the students above the
    previous plateau, (2) maximize how many students escape the new plateau, and
    pin that count. Stops at :data:`SATISFACTION_MAX` or when nobody escapes.
    Integer satisfaction makes both steps exact.

    Parameters
    ----------
    problem : model.CpSatProblem | model.CpSatSoftProblem
        The built model; mutated in place with the plateau constraints.
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


def _solve_stage(
    model: cp_model.CpModel,
    label: str,
    *,
    maximize: cp_model.LinearExprT | None = None,
    minimize: cp_model.LinearExprT | None = None,
) -> cp_model.CpSolver:
    """Optimize the given expression to proven optimality.

    Exactly one of ``maximize``/``minimize`` is given by the caller.

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
    SolverError
        If the stage does not reach proven optimality; a non-optimal stage
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
    if status != cp_model.OPTIMAL:
        raise SolverError(
            f"CP-SAT stage {label!r} ended with status "
            f"{solver.StatusName(status)!r}, not proven optimal"
        )
    return solver


def _extract(problem, solver: cp_model.CpSolver, preferences) -> CpSatSolution:
    """Read the solved values; satisfaction is recomputed in float per student.

    Parameters
    ----------
    problem : model.CpSatProblem | model.CpSatSoftProblem
        The built model, for the ``in_group``/``satisfied``/``satisfaction``
        variables to read back.
    solver : cp_model.CpSolver
        The solver holding the final stage's proven-optimal solution.
    preferences : pandas.DataFrame
        Long-format preference rows, indexed by ``(student, TypeWens, Nr)``.

    Returns
    -------
    CpSatSolution
        The solved assignment, honored wishes and recomputed satisfaction.
    """
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


def _float_satisfaction(preferences, satisfied: dict, students: list) -> dict:
    """Per-student float satisfaction from the honored wishes.

    The float twin of the model's integer element table: F(weighted honored
    sum), normalized by F(best case) when the student has positive wishes, or
    added to the baseline 1 when not.

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
