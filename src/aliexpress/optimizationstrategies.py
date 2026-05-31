"""Implement different strategies to optimize a set of scores into one number"""

import logging

import pulp

from . import preferences_utils

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


# Solving lexmaxmin level by level needs this many locals/branches; splitting it would
# obscure the iterative flow. Refactor tracked as a follow-up.
# pylint: disable=too-many-locals,too-many-branches
def plateaud_lexmaxmin(
    scores: dict[str, pulp.LpVariable],
    prob: pulp.LpProblem,
    n_levels_max: int = None,
    satisfaction_max: float = None,
    solver=None,
):
    """
    Solve the approximate lexmaxmin problem for scores

    Uses an iterative solve, making use of the fact that scores are often  plateaud:
    there are multiple scores at the same level. Level by level,
    first the next lowest plateau is determined, and then the number of values
    on that plateau. When each number is found, it is then added as a constraint and
    continues solving. Automatically stops when all students are distributed,
    or if n_levels max or satisfaction_max is reached. In that case,
    total score is the ultimate tie breaker.

    Parameters
    ----------
    scores : dict[str, pulp.LpVariable]
        The variables which should be optimized
    prob : pulp.LpProblem
        The problem to which the constraints are added
    n_levels_max : int, optional
        The max number of plateaus to use. Higher means more precision, but slightly slower,
        although the last levels are usually very quick, when the solution is already
        fixed.
    satisfaction_max : float, optional
        The satisfaction after which the relative satisfaction will be used. This prevents
        some numerical solver errors.
    solver : optional
        The pulp solver, which is needed because LexMaxMin requires solving the problem at
        each level
    """
    M = 100
    eps = 1e-5  # precision
    delta = 1e-4  # step size between plateaus
    solver = solver or pulp.PULP_CBC_CMD()
    satisfaction_max = satisfaction_max or float("inf")
    level = 0
    while True:
        if n_levels_max is not None and level >= n_levels_max:
            break
        # Step 1: maximize minimal satisfaction = determine next plateau
        minimal_score = pulp.LpVariable(f"MinimalScore_{level}")
        # pylint: disable=used-before-assignment
        if level == 0:
            for satisfaction in scores.values():
                prob += minimal_score <= satisfaction
        else:
            prob += minimal_score >= m_val + eps
            for student, satisfaction in scores.items():
                prob += (
                    minimal_score
                    <= satisfaction + (1 - has_this_level[student]) * M + eps
                ), f"MinimalSatisfactionLT{student}_{level}"
        # pylint: enable=used-before-assignment
        prob.sense = pulp.LpMaximize
        prob.setObjective(minimal_score)
        prob.solve(solver)
        m_val = minimal_score.value()
        logger.debug("Level %s, step 1 done, %s", level, m_val)

        if m_val > satisfaction_max:
            logger.debug("Minimal satisfaction reached, breaking lexmaxmin")
            break

        # Add as constraint
        if level == 0:
            for key in scores:
                prob += scores[key] >= m_val
        else:
            for key in scores:
                prob += (
                    scores[key] >= m_val * has_this_level[key] - eps
                ), f"MinimalSatisfaction_{key}_{level}"

        # Useful for debugging - usually from numerical errors
        # if level > 0:
        #     self.prob.solve(solver)

        # Step 2: minimize its occurrence
        has_this_level = pulp.LpVariable.dicts(
            f"HasThisLevel_{level}", scores.keys(), cat="Binary"
        )
        for key, value in scores.items():
            preferences_utils.apply_threshold_constraint(
                prob, value, m_val + delta, has_this_level[key], M=100
            )

        prob.sense = pulp.LpMaximize
        prob.setObjective(pulp.lpSum(has_this_level.values()))
        prob.solve(solver)
        count_at_level = sum(
            1 for key in scores if pulp.value(has_this_level[key]) > 0.5
        )
        logger.debug("Level %s, step 2 done, %s at this level", level, count_at_level)
        if count_at_level == 0:
            logger.debug("Stopped at level %s: no more students left", level)
            break
        # Add as constraint
        prob += pulp.lpSum(has_this_level.values()) == count_at_level
        level += 1

    return pulp.lpSum(scores.values())
