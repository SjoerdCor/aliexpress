"""From per-student satisfaction to a single objective value.

Given a satisfaction score per student, this module answers one question: what aggregate
value does the optimizer maximize? Three strategies are implemented:

- ``total``: sum of all satisfaction scores.
- ``lowest_score``: the minimum satisfaction, with total as tie-breaker.
- ``plateaud_lexmaxmin``: approximate lexicographic max-min, level by level.

Boundary with ``satisfaction.py``: that module maps honored preferences to a satisfaction
score per student (the *metric*). This module aggregates those scores across students into
a single objective (the *strategy*).
"""

import logging
import time

import pulp

from ..errors import SolverError
from . import pulp_thresholds
from ._balance import get_solver

logger = logging.getLogger(__name__)


def total(scores: dict[str, pulp.LpVariable]) -> pulp.LpVariable:
    """Optimize for the total of scores"""
    return pulp.lpSum(scores.values())


def lowest_score(
    scores: dict[str, pulp.LpVariable], prob: pulp.LpProblem
) -> pulp.LpVariable:
    """Optimize lowest score, with total score as tie-breaker

    A very basic/first-step (and therefore much quicker) version of lexmaxmin,
    which doesn't require intermediate solving the problem
    """

    minimal_score = pulp.LpVariable("MinimalScore")
    for satisfaction in scores.values():
        prob += minimal_score <= satisfaction
    M = 1_000_000
    return M * minimal_score + pulp.lpSum(scores.values())


# A stateful solver with a single entry point (solve); the level-by-level steps share
# evolving state (m_val, has_this_level) as instance attributes, which is exactly what a
# class is for here.
# pylint: disable=too-few-public-methods
class _PlateaudLexMaxMin:
    """Approximate lexmaxmin solve for scores, level by level.

    Makes use of the fact that scores are often plateaud: there are multiple scores at
    the same level. Per level, first the next lowest plateau is determined (by maximizing
    the minimal score), then the number of scores on that plateau is fixed, after which it
    continues with the next level. Stops when all scores are fixed, or when ``n_levels_max``
    or ``satisfaction_max`` is reached; the total score is the ultimate tie breaker.

    Parameters
    ----------
    scores : dict[str, pulp.LpVariable]
        The variables which should be optimized.
    prob : pulp.LpProblem
        The problem the level-by-level constraints are added to.
    solver : optional
        The pulp solver used to re-solve the problem at each level. Defaults to CBC.
    """

    EPS = 1e-5  # numerical precision
    DELTA = 1e-4  # step size between plateaus
    BIG_M = 100  # big-M for the threshold constraints

    def __init__(self, scores, prob, solver):
        self.scores = scores
        self.prob = prob
        self.solver = solver or pulp.PULP_CBC_CMD()
        # Carried from the previous level into the next.
        self.m_val = None
        self.has_this_level = None

    def solve(self, n_levels_max=None, satisfaction_max=None) -> pulp.LpVariable:
        """Run the level-by-level optimization, adding constraints to the problem.

        Parameters
        ----------
        n_levels_max : int, optional
            Max number of plateaus to process. Higher is more precise but slightly slower;
            later levels are usually quick once the solution is mostly fixed. No limit by
            default.
        satisfaction_max : float, optional
            Stop once the minimal score exceeds this; prevents some numerical solver errors
            at high satisfaction. No limit by default.

        Returns
        -------
        pulp.LpVariable
            The total-score expression, used as the ultimate tie breaker by the caller.
        """
        if satisfaction_max is None:
            satisfaction_max = float("inf")
        level = 0
        while True:
            if n_levels_max is not None and level >= n_levels_max:
                break
            t0 = time.perf_counter()
            self.m_val = self._raise_minimal_score(level)
            logger.debug("Level %s, step 1 done, %s", level, self.m_val)
            if self.m_val > satisfaction_max:
                logger.debug("Minimal satisfaction reached, breaking lexmaxmin")
                break
            self._fix_plateau_as_lower_bound(level)

            count_at_level = self._count_on_plateau(level)
            elapsed = time.perf_counter() - t0
            logger.debug(
                "Level %s, step 2 done, %s at this level", level, count_at_level
            )
            logger.info(
                "lexmaxmin level %d: Optimal in %.2fs, plateau=%d",
                level,
                elapsed,
                count_at_level,
            )
            if count_at_level == 0:
                logger.debug("Stopped at level %s: no more students left", level)
                break
            self.prob += pulp.lpSum(self.has_this_level.values()) == count_at_level
            level += 1

        return pulp.lpSum(self.scores.values())

    def _solve_and_check(self, level: int) -> None:
        """Solve the sub-problem and raise SolverError if the result is not Optimal.

        Raises
        ------
        SolverError
            If the solve does not reach optimality, naming the level and actual status.
            Without this check a non-optimal solve leaves variable values as ``None``,
            which causes a cryptic ``TypeError`` downstream.
        """
        self.prob.solve(self.solver)
        status = pulp.LpStatus[self.prob.status]
        if status != "Optimal":
            raise SolverError(
                f"Lexmaxmin sub-solve at level {level} did not reach optimality "
                f"(status: {status!r})"
            )

    def _raise_minimal_score(self, level: int) -> float:
        """Maximize the minimal score to find the next plateau ``level``; return its value."""
        minimal_score = pulp.LpVariable(f"MinimalScore_{level}")
        if level == 0:
            for satisfaction in self.scores.values():
                self.prob += minimal_score <= satisfaction
        else:
            self.prob += minimal_score >= self.m_val + self.EPS
            for student, satisfaction in self.scores.items():
                self.prob += (
                    minimal_score
                    <= satisfaction
                    + (1 - self.has_this_level[student]) * self.BIG_M
                    + self.EPS
                ), f"MinimalSatisfactionLT{student}_{level}"
        self.prob.sense = pulp.LpMaximize
        self.prob.setObjective(minimal_score)
        self._solve_and_check(level)
        return minimal_score.value()

    def _fix_plateau_as_lower_bound(self, level: int) -> None:
        """Lock in the plateau found at ``level`` as a lower bound for every score."""
        if level == 0:
            for key in self.scores:
                self.prob += self.scores[key] >= self.m_val
        else:
            for key in self.scores:
                self.prob += (
                    self.scores[key] >= self.m_val * self.has_this_level[key] - self.EPS
                ), f"MinimalSatisfaction_{key}_{level}"

    def _count_on_plateau(self, level: int) -> int:
        """Maximize, then count, how many scores sit on plateau ``level``."""
        self.has_this_level = pulp.LpVariable.dicts(
            f"HasThisLevel_{level}", self.scores.keys(), cat="Binary"
        )
        for key, value in self.scores.items():
            pulp_thresholds.apply_threshold_constraint(
                self.prob,
                value,
                self.m_val + self.DELTA,
                self.has_this_level[key],
                M=self.BIG_M,
            )
        self.prob.sense = pulp.LpMaximize
        self.prob.setObjective(pulp.lpSum(self.has_this_level.values()))
        self._solve_and_check(level)
        return sum(
            1 for key in self.scores if pulp.value(self.has_this_level[key]) > 0.5
        )


def plateaud_lexmaxmin(
    scores: dict[str, pulp.LpVariable],
    prob: pulp.LpProblem,
    n_levels_max: int = None,
    satisfaction_max: float = None,
    solver=None,
) -> pulp.LpVariable:
    """Solve the approximate lexmaxmin problem for scores.

    Thin wrapper around :class:`_PlateaudLexMaxMin`; see that class and its ``solve`` for the
    algorithm and parameter descriptions. Returns the total-score tie-breaker expression.
    """
    return _PlateaudLexMaxMin(scores, prob, solver).solve(
        n_levels_max=n_levels_max, satisfaction_max=satisfaction_max
    )


def set_optimization_target(solver, studentsatisfaction: dict) -> None:
    """Set the objective of ``solver.prob`` according to ``solver.optimize``.

    Dispatches to the appropriate aggregation strategy and adds the resulting expression as
    the LP objective. Valid values for ``solver.optimize``: ``"studentsatisfaction"``
    (total), ``"least_satisfied"`` (min with total tie-breaker), ``"lexmaxmin"``
    (plateaud lexicographic max-min).

    Parameters
    ----------
    solver:
        A :class:`~aliexpress.solver.problemsolver.ProblemSolver` instance.
    studentsatisfaction:
        Dict of ``{student: pulp.LpVariable}`` for per-student satisfaction.
    """
    if solver.optimize == "studentsatisfaction":
        optimization_target = total(studentsatisfaction)
    elif solver.optimize == "least_satisfied":
        optimization_target = lowest_score(studentsatisfaction, solver.prob)
    elif solver.optimize == "lexmaxmin":
        optimization_target = plateaud_lexmaxmin(
            studentsatisfaction,
            solver.prob,
            satisfaction_max=0.8,
            solver=get_solver(),
        )
    else:
        raise ValueError(f"Unknown optimization strategy {solver.optimize!r}")
    solver.prob += optimization_target
