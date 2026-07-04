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

from aliexpress.data import preferences_data
from aliexpress.data.datareader import matching_key
from aliexpress.main import _read_groups, _read_preferences
from aliexpress.solver._balance import GroupBalance
from aliexpress.solver.cpsat import engine
from aliexpress.solver.cpsat.results import to_solution_result


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


def test_total_strategy_maximizes_sum_over_lexmaxmin():
    """`optimize="total"` never scores a lower satisfaction sum than `"lexmaxmin"`.

    `"total"` maximizes the satisfaction sum directly, with no plateau
    constraints; `"lexmaxmin"` raises the same sum as a tie-break only after
    fixing the fairness plateaus, which can trade sum for a higher minimum. So
    the sum under `"total"` is always at least the sum under `"lexmaxmin"`.
    """
    preference_data, target_groups, not_together = _small_instance()
    balance = GroupBalance(max_imbalance_boys_girls_total=7)

    total_solution = engine.solve_with_fixed_balance(
        preferences=preference_data.preferences,
        students=preference_data.students_info,
        groups_to=target_groups.counts,
        not_together=not_together,
        groupbalance=balance,
        optimize="total",
    )
    lexmaxmin_solution = engine.solve_with_fixed_balance(
        preferences=preference_data.preferences,
        students=preference_data.students_info,
        groups_to=target_groups.counts,
        not_together=not_together,
        groupbalance=balance,
        optimize="lexmaxmin",
    )

    total_sum = round(sum(total_solution.student_satisfaction.values()), 6)
    lexmaxmin_sum = round(sum(lexmaxmin_solution.student_satisfaction.values()), 6)
    assert total_sum >= lexmaxmin_sum

    assert set(total_solution.assignment) == set(preference_data.students_info)
    assert all(
        group in target_groups.counts for group in total_solution.assignment.values()
    )


def test_solution_result_matches_cpsat_solution():
    """`to_solution_result` derives a consistent `SolutionResult` from a solved instance."""
    preference_data, target_groups, not_together = _small_instance()

    solution = engine.solve_with_fixed_balance(
        preferences=preference_data.preferences,
        students=preference_data.students_info,
        groups_to=target_groups.counts,
        not_together=not_together,
        groupbalance=GroupBalance(max_imbalance_boys_girls_total=7),
    )
    result = to_solution_result(
        solution,
        preference_data.preferences,
        preference_data.students_info,
        target_groups.counts,
    )

    graag_met = preferences_data.get_graag_met(preference_data.preferences)
    assert result.weights == dict(graag_met["Gewicht"])

    for key, weight in result.weights.items():
        satisfied = result.satisfied[key]
        expected = satisfied * weight if weight > 0 else (1 - satisfied) * weight
        assert result.weighted_satisfied[key] == expected

    assert result.student_satisfaction == solution.student_satisfaction

    _assert_group_composition_reconciles(result, preference_data, target_groups)


def _assert_group_composition_reconciles(result, preference_data, target_groups):
    """The mapped `group_composition` must match a plain recount from `assignment`."""
    boys_in_group: dict[str, int] = {group: 0 for group in target_groups.counts}
    girls_in_group: dict[str, int] = {group: 0 for group in target_groups.counts}
    for student, group in result.assignment.items():
        if preference_data.students_info[student]["Jongen/meisje"] == "Jongen":
            boys_in_group[group] += 1
        else:
            girls_in_group[group] += 1

    for group, occupancy in target_groups.counts.items():
        composition = result.group_composition[group]
        assert composition.boys_year == boys_in_group[group]
        assert composition.girls_year == girls_in_group[group]
        assert composition.boys_total == occupancy["Jongens"] + boys_in_group[group]
        assert composition.girls_total == occupancy["Meisjes"] + girls_in_group[group]


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
