"""From honored preferences to satisfaction per student.

This module defines the *metric* that maps honored preferences to a satisfaction score
per student. Conceptually this is one pluggable choice — two concrete metrics are
provided:

- ``get_satisfaction_integral``: concave, diminishing-returns scoring. The marginal value
  of each additional honored preference decreases. A lexmaxmin objective built on top of
  it then drives the "everyone gets preference 1 first" property, not this function alone.
- ``get_satisfaction_percentage``: linear scoring. Treats all preferences as equally
  important regardless of order.

Narrative through the file:
  honored preferences → achievable weighted levels → satisfaction function → score per student
"""

import itertools

from ..data import preferences_data

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


def _normalize_and_bound(weighted: float, best: float, worst: float) -> float:
    """Normalize a student's weighted honored sum to a satisfaction score.

    Parameters
    ----------
    weighted:
        The student's weighted honored sum: an honored positive wish contributes its
        weight, a violated negative (avoid) wish contributes its (negative) weight, and
        anything else contributes 0.
    best:
        Sum of the student's positive weights — the maximum achievable ``weighted``.
    worst:
        Sum of the student's negative weights — the minimum achievable ``weighted``
        (always ``<= 0``).

    Returns
    -------
        A satisfaction score:

        - If the student has any positive wishes (``best > 0``, including students with
          both positive and negative wishes), the score is
          ``get_satisfaction_integral(0, weighted) / get_satisfaction_integral(0, best)``
          — unchanged from the original behavior. This branch is not bounded below at -1:
          a mixed student whose avoid-wishes are violated can score well under -100%.
        - Otherwise, if the student only has avoid-wishes and at least one is violated
          (``worst < 0`` and ``weighted != 0``), the score is normalized by
          ``abs(get_satisfaction_integral(0, worst))`` instead: a single violated
          avoid-wish always scores -1.0 regardless of its weight, and further violations
          saturate towards -1.0 rather than diverging.
        - Otherwise (nothing was violated, or the student had no wishes at all), the
          score is the baseline 1.0. This also captures the "jump" from 0 to +1.0 when a
          student with only avoid-wishes is kept away from everyone they wanted to avoid.
    """
    raw = get_satisfaction_integral(0, weighted)
    if best > 0:
        return raw / get_satisfaction_integral(0, best)
    if worst < 0 and weighted != 0:
        return raw / abs(get_satisfaction_integral(0, worst))
    return 1.0


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
