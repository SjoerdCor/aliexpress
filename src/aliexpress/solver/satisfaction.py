"""Helper functions from problemsolver, to determine preferences"""

import itertools
import warnings

import pandas as pd
import pulp

from .. import preferences_data
from . import pulp_thresholds


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def _all_unique_sums(iterable):
    """Calculate all possible sums from sublists from the iterable"""
    return {sum(l) for l in powerset(iterable)}


def get_possible_weighted_preferences(preferences) -> set:
    """
    Get all the possible number of weighted preferences

    This will be used to know for which values a satisfaction score must be calculated
    and which dictionary values must be calculated per student. By minimizing this number,
    we make the problem calculation as fast as possible, while allowing for arbitrary precision

    Parameters
    ----------
    preferences: pd.DataFrame
        The DataFrame containing the preferences of the students, must have a MultiIndex
        with levels ("Leerling", "TypeWens") with columns ("Waarde" & "Gewicht")
    """
    unique_weighted_preferences_per_student = (
        preferences_data.get_graag_met(preferences)
        .groupby("Leerling")["Gewicht"]
        .apply(_all_unique_sums)
    )

    unique_weighted_preferences = set()
    for wp in unique_weighted_preferences_per_student:
        unique_weighted_preferences.update(wp)
    return unique_weighted_preferences


def get_satisfaction_integral(x_a: float, x_b: float) -> float:
    """
    Calculate the extra satisfaction from granting x_b preferences instead of x_a

    This is the (scaled) integral of 0.5**x. This satisfaction function ensures everybody
    first gets their first preference, then everybody their second preference, etc.

    Parameters
    ----------
    x_a: float
        The number of (weighted) preferences as the basic satisfaction of the student
    x_b: float
        The number of (weighted) preferences as the goal satisfaction of the student

    Returns
    -------
        The added satisfaction score of the student
    """
    # In principle, we should probably only specify the satisfaction function and
    # then have this just be a numerical integration for optimal flexibility, but since
    # this flexibility isn't required yet, we're using a analytical integration.
    return (-(0.5**x_b)) - (-(0.5**x_a))


def calculate_added_satisfaction(preferences) -> dict:
    """
    Calculate the score of getting all possible weighted preferences values accounted for
    """

    possible_weighted_preferences = get_possible_weighted_preferences(preferences)

    # Sorting is important since we're going to difference!
    positive_values = sorted(v for v in possible_weighted_preferences if v >= 0)
    negative_values = sorted(
        (v for v in possible_weighted_preferences if v <= 0), reverse=True
    )

    preference_value = {}
    for values in (negative_values, positive_values):
        # The 0 value is deliberately not taken into account!
        # This would lead to ZeroDivisionErrors
        for last_wp, wp in zip(values[:-1], values[1:]):
            preference_value[wp] = get_satisfaction_integral(last_wp, wp)
    return preference_value


def calculate_weighted_preferences(
    solver, satisfied: dict, prob: pulp.LpProblem
) -> dict:
    """Calculate the weighted sum of satisfied preferences and add LP variables to ``prob``."""
    graag_met = preferences_data.get_graag_met(solver.preferences)
    weights = graag_met["Gewicht"].to_dict()
    weights_pulp = pulp.LpVariable.dicts(
        "Weights_preferences", graag_met.index.to_list(), cat="Continuous"
    )
    weighted_satisfied = pulp.LpVariable.dicts(
        "WeightedSatisfied", graag_met.index.to_list(), cat="Continuous"
    )

    for key, weight in weights.items():
        prob += weights_pulp[key] == weight
        if weight > 0:
            # Weight is positive: you get points for getting it right
            prob += weighted_satisfied[key] == (satisfied[key] * weight)
        else:
            # Weight is negative: you get deduction if you do it wrong
            prob += weighted_satisfied[key] == ((1 - satisfied[key]) * weight)

    return weighted_satisfied


def calculate_student_satisfaction(
    solver, satisfied: dict, prob: pulp.LpProblem
) -> dict:
    """Compute per-student satisfaction variables and add them to ``prob``."""
    added_satisfaction = calculate_added_satisfaction(solver.preferences)
    weighted_satisfied = calculate_weighted_preferences(solver, satisfied, prob)

    for student in solver.students:
        student_weighted = [
            weighted_satisfied.get((student, i), 0)
            for i in range(1, len(added_satisfaction) + 1)
        ]
        wp_satisfied = pulp.lpSum(student_weighted)

        wp_satisfied_per_student = pulp.LpVariable.dicts(
            f"{student}_weighted_preferences_accountend",
            added_satisfaction.keys(),
            cat="Binary",
        )

        pulp_thresholds.apply_threshold_constraints(
            prob,
            wp_satisfied,
            added_satisfaction.keys(),
            wp_satisfied_per_student,
            eps=1e-3,  # Necessary to run lexmaxmin without errors; I dont know why
        )

        satisfaction_current_student = pulp.lpSum(
            val * wp_satisfied_per_student[n_wp]
            for n_wp, val in added_satisfaction.items()
        )

        with warnings.catch_warnings(
            action="ignore", category=pd.errors.PerformanceWarning
        ):
            # Add base satisfaction if no (positive) preferences, so maxmin optimizes
            # for student with actual preferences
            try:
                preferences = solver.preferences.loc[(student, "Graag met")]
            except KeyError:
                satisfaction_current_student = 1
            else:
                positive_preferences = preferences.query("Gewicht > 0")
                if positive_preferences.empty:
                    satisfaction_current_student += 1
                else:
                    max_wishes = positive_preferences["Gewicht"].sum()
                    max_satisfaction = get_satisfaction_integral(0, max_wishes)
                    satisfaction_current_student /= max_satisfaction
        prob += solver.studentsatisfaction[student] == satisfaction_current_student
    return solver.studentsatisfaction
