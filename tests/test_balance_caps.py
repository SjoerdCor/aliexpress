"""Tests that a BalanceMaxima caps the per-family slack domain in the model.

The capping arithmetic is tested directly on ``_slack_upper`` (no CP-SAT model),
and a single wiring test confirms ``build_soft_problem`` threads a ``maxima``
through to the actual slack domains.
"""

import pandas as pd
from ortools.sat.python import cp_model

from aliexpress.solver._balance import BalanceMaxima
from aliexpress.solver._balance_families import (
    STRICTEST_LIMIT,
    _slack_upper,
    uncapped_slack_bound,
)
from aliexpress.solver.modelbuilder import build_soft_problem


def test_slack_upper_caps_named_family():
    """A capped family's slack tops out at its cap minus STRICTEST_LIMIT."""
    upper = _slack_upper(
        "diff_total", BalanceMaxima(max_diff_n_students_total=3), uncapped_bound=10
    )
    assert upper == 3 - STRICTEST_LIMIT  # cap 3 - 1 = 2


def test_slack_upper_none_maxima_uses_default():
    """maxima=None leaves the slack at its uncapped bound."""
    assert _slack_upper("diff_total", None, uncapped_bound=10) == 10


def test_slack_upper_none_field_uses_default():
    """A None field within a populated BalanceMaxima leaves that family uncapped."""
    maxima = BalanceMaxima(max_clique=5)
    assert _slack_upper("diff_total", maxima, uncapped_bound=10) == 10


def test_slack_upper_cap_equal_to_strictest_limit_pins_family():
    """A cap equal to STRICTEST_LIMIT pins the family (slack upper bound 0)."""
    upper = _slack_upper(
        "diff_total",
        BalanceMaxima(max_diff_n_students_total=STRICTEST_LIMIT),
        uncapped_bound=10,
    )
    assert upper == 0  # pins the family at STRICTEST_LIMIT


def _wiring_scenario():
    """A small solvable instance with one real 'Graag met' preference row.

    Four students over three empty groups gives ``uncapped_slack_bound`` == 4, above
    the cap under test (3), so a capped family's slack reaches a visibly lower
    maximum than the uncapped default. One positive preference keeps the
    preferences frame non-empty (the empty frame is an unsupported edge case).
    """
    preferences = pd.DataFrame(
        [
            {
                "Leerling": "s0",
                "TypeWens": "Graag met",
                "Nr": 1.0,
                "Waarde": "s1",
                "Gewicht": 1.0,
            }
        ]
    ).set_index(["Leerling", "TypeWens", "Nr"])
    students = {
        f"s{i}": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Jongen" if i % 2 else "Meisje",
            "Stamgroep": "a",
        }
        for i in range(4)
    }
    groups_to = {
        group: {"Jongens": 0, "Meisjes": 0} for group in ("blauw", "geel", "rood")
    }
    return preferences, students, groups_to


def _max_reachable_slack(problem, family) -> int:
    """The largest value the family's slack can take, read via a solve.

    Maximising the slack alone drives it to its domain upper bound (a looser
    balance limit is always feasible), so the solved value equals that bound.
    Read through ``solver.Value`` rather than ``var.Proto().domain``: the proto
    reflection on a freshly built (unsolved) CP-SAT variable corrupts the heap
    on this platform (the same fragility documented in
    ``modelbuilder._add_satisfaction``), whereas the solved value is safe.
    """
    problem.model.Maximize(problem.slacks[family])
    solver = cp_model.CpSolver()
    status = solver.Solve(problem.model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return solver.Value(problem.slacks[family])


def test_build_soft_problem_caps_only_the_named_family():
    """build_soft_problem caps the named family's slack, leaving others uncapped."""
    preferences, students, groups_to = _wiring_scenario()
    maxima = BalanceMaxima(max_diff_n_students_total=3)

    # Capped family tops out at cap - STRICTEST_LIMIT; a separate build for the
    # uncapped family, since each solve fixes its own single objective.
    capped = build_soft_problem(preferences, students, groups_to, [], maxima=maxima)
    uncapped = build_soft_problem(preferences, students, groups_to, [], maxima=maxima)

    assert _max_reachable_slack(capped, "diff_total") == 3 - STRICTEST_LIMIT
    assert _max_reachable_slack(uncapped, "gender_total") == uncapped_slack_bound(
        students, groups_to
    )


def test_build_soft_problem_without_maxima_leaves_family_uncapped():
    """Without a maxima, every family keeps its uncapped slack bound."""
    preferences, students, groups_to = _wiring_scenario()

    problem = build_soft_problem(preferences, students, groups_to, [])

    assert _max_reachable_slack(problem, "diff_total") == uncapped_slack_bound(
        students, groups_to
    )
