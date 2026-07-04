"""Meetpoort: the CP-SAT engine must reproduce the pinned pulp/HiGHS optimum.

The per-student satisfaction values in ``test_integration_main`` are the uniquely
determined optimization objective; they are solver-independent by design. These tests
run the CP-SAT pipeline on the same instances and compare against the same pinned
values — the equivalence guard for the backend migration.
"""

import time

from test_integration_main import (
    _FULL_SATISFACTION,
    _NOT_TOGETHER_FULL,
    _NOT_TOGETHER_SMALL,
    _SMALL_SATISFACTION,
)

from aliexpress.data.datareader import matching_key
from aliexpress.main import _read_groups, _read_preferences
from aliexpress.solver._balance import GroupBalance
from aliexpress.solver.cpsat import engine


def _small_instance():
    """The small doorzetten instance, read exactly as production reads it."""
    target_groups = _read_groups("tests/integration/groepen_small.xlsx")
    preference_data = _read_preferences(
        "tests/integration/voorkeuren_small.xlsx", target_groups.counts
    )
    not_together = [
        {**rule, "group": {matching_key(s) for s in rule["group"]}}
        for rule in _NOT_TOGETHER_SMALL
    ]
    return preference_data, target_groups, not_together


def test_small_instance_reproduces_pinned_satisfaction():
    """Manual-balance path: same optimum as pulp/HiGHS, per student, to 1e-6."""
    preference_data, target_groups, not_together = _small_instance()

    solution = engine.solve_with_fixed_balance(
        preferences=preference_data.preferences,
        students=preference_data.students_info,
        groups_to=target_groups.counts,
        not_together=not_together,
        groupbalance=GroupBalance(max_imbalance_boys_girls_total=7),
    )

    expected = {
        matching_key(name): value
        for name, value in _SMALL_SATISFACTION["Tevredenheid"].items()
    }
    actual = {
        student: round(value, 6)
        for student, value in solution.student_satisfaction.items()
    }
    assert actual == expected


def _full_instance():
    """The full production-scale instance, read exactly as production reads it."""
    target_groups = _read_groups("tests/integration/groepen.xlsx")
    preference_data = _read_preferences(
        "tests/integration/voorkeuren.xlsx", target_groups.counts
    )
    not_together = [
        {**rule, "group": {matching_key(s) for s in rule["group"]}}
        for rule in _NOT_TOGETHER_FULL
    ]
    return preference_data, target_groups, not_together


def test_full_instance_reproduces_pinned_satisfaction():
    """Automatic-balance path: same optimum as pulp/HiGHS, per student, to 1e-6.

    The pulp/HiGHS reference for this instance takes ~98s; this asserts the
    CP-SAT wall time stays well under the 2-minute measurement-gate limit.
    """
    preference_data, target_groups, not_together = _full_instance()

    start = time.perf_counter()
    solution = engine.solve_within_minimal_relaxation(
        preferences=preference_data.preferences,
        students=preference_data.students_info,
        groups_to=target_groups.counts,
        not_together=not_together,
    )
    elapsed = time.perf_counter() - start
    print(f"CP-SAT full-instance solve: {elapsed:.2f}s")
    assert elapsed < 120

    expected = {
        matching_key(name): value
        for name, value in _FULL_SATISFACTION["Tevredenheid"].items()
    }
    actual = {
        student: round(value, 6)
        for student, value in solution.student_satisfaction.items()
    }
    assert actual == expected
