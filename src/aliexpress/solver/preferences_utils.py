"""Helper functions from problemsolver, to determine preferences"""

import itertools

import pandas as pd


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def _all_unique_sums(iterable):
    """Calculate all possible sums from sublists from the iterable"""
    return {sum(l) for l in powerset(iterable)}


def get_graag_met(preferences: pd.DataFrame) -> pd.DataFrame:
    """Return the 'Graag met' slice of preferences; empty DataFrame when none present.

    Equivalent to preferences.xs("Graag met", level="TypeWens") but safe when
    no positive preferences exist.
    """
    mask = preferences.index.get_level_values("TypeWens") == "Graag met"
    return preferences.loc[mask].droplevel("TypeWens")


def get_possible_weighted_preferences(preferences: pd.DataFrame) -> set:
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
        get_graag_met(preferences)
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
