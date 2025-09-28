"""Implement different strategies to optimize a set of scores into one number"""

import logging

import pulp

from aliexpress import preferences_utils

logger = logging.getLogger(__name__)


def total(scores: dict[str, pulp.LpVariable]) -> pulp.LpVariable:
    """Optimize for the total of scores"""
    return pulp.lpSum(scores.values())


def lowest_score(
    scores: dict[str, pulp.LpVariable], prob: pulp.LpProblem
) -> pulp.LpVariable:
    """Optimize lowest score, with total satisfaction as tie-breaker

    A very basic/first-step (and therefore much quicker) version of lexmaxmin
    """

    minimal_satisfaction = pulp.LpVariable("MinimalSatisfaction")
    for satisfaction in scores.values():
        prob += minimal_satisfaction <= satisfaction
    M = 1_000_000
    return M * minimal_satisfaction + pulp.lpSum(scores.values())


def plateaud_lexmaxmin(
    scores: dict[str, pulp.LpVariable],
    prob: pulp.LpProblem,
    n_levels_max: int = None,
    satisfaction_max: float = None,
    solver=None,
):
    """
    Solve the approximate lexmaxmin problem for student satisfaction

    Uses an iterative solve, making use of the fact that student satisfaction is
    often plateaud: there are multiple students at the same level. Level by level,
    first the next lowest plateau is determined, and then the number of students
    on that plateau. When each number is found, it is then added as a constraint and
    continues solving. Automatically stops when all students are distributed,
    or if n_levels max or satisfaction_max is reached. In that case,
    totalstudent satisfaction is the ultimate tie breaker.

    Parameters
    ----------
    n_levels_max : int, optional
        The max number of plateaus to use. Higher means more precision, but slightly slower,
        although the last levels are usually very quick, when the solution is already
        fixed.
    satisfaction_max : float (default 0.8)
        The satisfaction after which the relative satisfaction will be used. This prevents
        some numerical solver errors.
    """
    M = 100
    eps = 1e-6
    solver = solver or pulp.PULP_CBC_CMD()

    level = 0
    while True:
        if n_levels_max is not None and level >= n_levels_max:
            break
        # Step 1: maximize minimal satisfaction = determine next plateau
        minimal_satisfaction = pulp.LpVariable(f"MinimalSatisfaction_{level}")
        # pylint: disable=used-before-assignment
        if level == 0:
            for satisfaction in scores.values():
                prob += minimal_satisfaction <= satisfaction
        else:
            prob += minimal_satisfaction >= m_val + eps
            for student, satisfaction in scores.items():
                prob += (
                    minimal_satisfaction
                    <= satisfaction + (1 - has_this_level[student]) * M + eps
                ), f"MinimalSatisfactionLT{student}_{level}"
        # pylint: enable=used-before-assignment
        prob.sense = pulp.LpMaximize
        prob.setObjective(minimal_satisfaction)
        prob.solve(solver)
        m_val = minimal_satisfaction.value()
        logger.debug("Level %s, step 1 done, %s", level, m_val)

        if m_val > satisfaction_max:
            logger.debug("Minimal satisfaction reached, breaking lexmaxmin")
            break

        # Add as constraint
        if level == 0:
            for student in scores:
                prob += scores[student] >= m_val
        else:
            for student in scores:
                prob += (
                    scores[student] >= m_val * has_this_level[student] - eps
                ), f"MinimalSatisfaction_{student}_{level}"

        # Useful for debugging - usually from numerical errors
        # if level > 0:
        #     self.prob.solve(solver)

        # Step 2: minimize its occurrence
        has_this_level = pulp.LpVariable.dicts(
            f"HasThisLevel_{level}", scores.keys(), cat="Binary"
        )
        delta = 1e-5
        for student in scores:
            has_this_level_student = pulp.LpVariable.dicts(
                f"HasLevel_{level}_{student}", [m_val + delta], cat="Binary"
            )
            preferences_utils.apply_threshold_constraints(
                prob,
                scores[student],
                [m_val + delta],
                has_this_level_student,
                M=100,
            )
            prob += has_this_level[student] == has_this_level_student[m_val + delta]
        prob.sense = pulp.LpMaximize
        prob.setObjective(pulp.lpSum(has_this_level.values()))
        prob.solve(solver)

        count_at_level = sum(
            1 for student in scores if pulp.value(has_this_level[student]) > 0.5
        )
        logger.debug("Level %s, step 2 done, %s", level, count_at_level)
        if count_at_level == 0:
            logger.debug("Stopped at level %s: no more students left", level)

            break
        # Add as constraint
        prob += pulp.lpSum(has_this_level.values()) == count_at_level
        level += 1

    return pulp.lpSum(scores.values())
