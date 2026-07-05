"""Integer scaling of preference weights for the CP-SAT model.

CP-SAT variables and coefficients are integers, but preference weights are
user-entered floats (the web form offers {0.5, 1, 2, 5}; the Excel route allows
any positive value). The model therefore works with ``weight * scale`` where
``scale`` is the smallest whole factor that makes every weight a whole number.

Contract: weights are treated as exact to 3 decimals. This covers every value a
user can meaningfully enter while absorbing binary float noise (0.1 + 0.2 scales
identically to 0.3). Under that contract the smallest exact scale is
``1000 / gcd(round(weight * 1000)..., 1000)`` — data-driven, so the form's weight
set costs only a factor 2 and whole weights are not scaled at all.
"""

import math
from typing import Iterable


def weight_scale(weights: Iterable[float]) -> int:
    """Return the smallest whole factor that makes every weight a whole number.

    The sign of a weight is irrelevant ("Liever niet met" weights are negative
    after negation); the scale is derived from the magnitudes.

    Raises
    ------
    ValueError
        If a weight rounds to 0 at the 3-decimal resolution: scaling would then
        silently erase that wish from the model.
    """
    thousandths = []
    for weight in weights:
        scaled = round(abs(weight) * 1000)
        if scaled == 0:
            raise ValueError(
                f"Weight {weight!r} is zero at the 3-decimal resolution; "
                "it cannot be represented in the integer model"
            )
        thousandths.append(scaled)
    return 1000 // math.gcd(*thousandths, 1000)
