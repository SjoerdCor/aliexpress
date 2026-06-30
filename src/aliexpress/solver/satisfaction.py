"""From honored preferences to satisfaction per student.

Boundaries
----------
- ``problemsolver`` owns the LP model: which preferences are honored given an assignment.
  It calls ``calculate_student_satisfaction`` to add the per-student satisfaction
  variables to the model.
- ``optimizationstrategies`` aggregates: given a satisfaction per student, what is the
  single objective value? That is a different question from the metric per student.

What this module does
---------------------
It defines the *metric* that maps honored preferences to a satisfaction score per student.
Conceptually this is one pluggable choice — two concrete metrics are provided:

- ``get_satisfaction_integral``: concave, diminishing-returns scoring. The marginal value
  of each additional honored preference decreases. The lexmaxmin objective in
  ``optimizationstrategies`` then drives the "everyone gets preference 1 first" property,
  not this function alone.
- ``get_satisfaction_percentage``: linear scoring. Treats all preferences as equally
  important regardless of order.

Narrative through the file:
  honored preferences → achievable weighted levels → satisfaction function → score per student
"""

import itertools
import warnings

import pandas as pd
import pulp

from ..data import preferences_data
from . import pulp_thresholds

# ---------------------------------------------------------------------------
# 1. Satisfaction functions (the pluggable metric)
# ---------------------------------------------------------------------------


def get_satisfaction_integral(x_a: float, x_b: float) -> float:
    """Extra satisfaction gained from x_a to x_b honored (weighted) preferences.

    Concave, diminishing-returns: the marginal gain of the n-th preference is less than
    that of the (n-1)-th. Computed as the integral of 0.5^x between x_a and x_b.

    Parameters
    ----------
    x_a:
        Current weighted level (lower bound of the integral).
    x_b:
        Target weighted level (upper bound).

    Returns
    -------
        The added satisfaction score between x_a and x_b.
    """
    # Closed-form integral of 0.5^x; more flexible numerical integration would change
    # nothing since the integrand is unlikely to change.
    return (-(0.5**x_b)) - (-(0.5**x_a))


def get_satisfaction_percentage(honored_weight: float, max_weight: float) -> float:
    """Fraction of the maximum weighted preferences that is honored (0.0 to 1.0).

    Linear alternative to ``get_satisfaction_integral``: treats all preferences as
    equally important regardless of order. A student with 2 of 4 weighted preferences
    honored scores 0.5, regardless of which two they got.

    Parameters
    ----------
    honored_weight:
        Sum of weights of the preferences that are honored.
    max_weight:
        Sum of all positive preference weights (the maximum achievable).

    Returns
    -------
        ``honored_weight / max_weight``, or 1.0 when max_weight is 0.
    """
    if max_weight == 0:
        return 1.0
    return honored_weight / max_weight


# ---------------------------------------------------------------------------
# 2. The achievable range (specific to the integral metric)
# ---------------------------------------------------------------------------


def _powerset(iterable):
    """All subsets of ``iterable`` as a generator of tuples."""
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def _all_unique_sums(iterable):
    """All possible sums of subsets of ``iterable``."""
    return {sum(subset) for subset in _powerset(iterable)}


def _achievable_weighted_levels(preferences) -> set:
    """All weighted preference levels reachable by at least one student.

    Determines which levels need a satisfaction score for the integral metric. Keeping
    this set minimal keeps the LP compact across arbitrary weight distributions.

    Parameters
    ----------
    preferences:
        Long-format DataFrame with MultiIndex ``(Leerling, TypeWens, Nr)`` and
        columns ``Waarde`` and ``Gewicht``.
    """
    unique_per_student = (
        preferences_data.get_graag_met(preferences)
        .groupby("Leerling")["Gewicht"]
        .apply(_all_unique_sums)
    )

    unique_levels: set = set()
    for wp in unique_per_student:
        unique_levels.update(wp)
    return unique_levels


def calculate_added_satisfaction(preferences) -> dict:
    """Marginal satisfaction score per achievable weighted level (integral metric).

    Returns ``{level: score}`` where ``score`` is the added satisfaction from the
    previous achievable level to ``level``, using ``get_satisfaction_integral``. Used as
    LP objective coefficients per student. Specific to the integral metric: the percentage
    metric does not need level-by-level coefficients.
    """
    possible_levels = _achievable_weighted_levels(preferences)

    # Sorting is important since we're going to difference!
    positive_values = sorted(v for v in possible_levels if v >= 0)
    negative_values = sorted((v for v in possible_levels if v <= 0), reverse=True)

    preference_value = {}
    for values in (negative_values, positive_values):
        # The 0 value is deliberately not taken into account!
        # This would lead to ZeroDivisionErrors
        for last_wp, wp in zip(values[:-1], values[1:]):
            preference_value[wp] = get_satisfaction_integral(last_wp, wp)
    return preference_value


# ---------------------------------------------------------------------------
# 3. Application to the LP
# ---------------------------------------------------------------------------


def calculate_weighted_preferences(
    solver, satisfied: dict, prob: pulp.LpProblem
) -> dict:
    """Add LP variables for weighted honored preferences to ``prob`` and return them.

    ``satisfied[key]`` is 1 when the preference is honored, 0 otherwise.
    """
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
            prob += weighted_satisfied[key] == (satisfied[key] * weight)
        else:
            # Negative weight → lower satisfaction when not satisfied (satisfied==0),
            # so (1 - satisfied) selects that case.
            prob += weighted_satisfied[key] == ((1 - satisfied[key]) * weight)

    return weighted_satisfied


def calculate_student_satisfaction(
    solver, satisfied: dict, prob: pulp.LpProblem
) -> dict:
    """Add per-student satisfaction LP variables to ``prob`` and return them.

    Applies the integral satisfaction metric to map each student's honored preferences
    to a single satisfaction variable. Students without positive preferences receive a
    baseline score so they do not drive the lexmaxmin objective.
    """
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
            # for students with actual preferences
            try:
                preferences = solver.preferences.loc[(student, "Graag met")]
            except KeyError:
                satisfaction_current_student = 1
            else:
                positive_preferences = preferences.query("Gewicht > 0")
                if positive_preferences.empty:
                    satisfaction_current_student += 1
                else:
                    max_preferences = positive_preferences["Gewicht"].sum()
                    max_satisfaction = get_satisfaction_integral(0, max_preferences)
                    satisfaction_current_student /= max_satisfaction
        prob += solver.studentsatisfaction[student] == satisfaction_current_student
    return solver.studentsatisfaction
