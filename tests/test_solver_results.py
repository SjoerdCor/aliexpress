"""Unit tests for solver/results.py: mapping a Solution into a SolutionResult.

Named apart from tests/test_results.py, which covers the unrelated web results blueprint.
"""

import pandas as pd

from aliexpress.solver.engine import Solution
from aliexpress.solver.results import SexCounts, to_solution_result


def _preferences(students: list[str]) -> pd.DataFrame:
    """One fulfilled 'Graag met' wish per student, weight 1.0."""
    index = pd.MultiIndex.from_tuples(
        [(s, "Graag met", 1) for s in students], names=["student", "TypeWens", "Nr"]
    )
    return pd.DataFrame({"Gewicht": [1.0] * len(students)}, index=index)


def _solution(assignment: dict[str, str]) -> Solution:
    students = list(assignment)
    return Solution(
        assignment=assignment,
        satisfied={(s, 1): True for s in students},
        student_satisfaction={s: 1.0 for s in students},
    )


def test_per_year_counts_split_by_jaarlaag():
    """Two jaarlagen assigned to the same group get separate per_year cohorts."""
    assignment = {"Anna": "A", "Bram": "A", "Cas": "A", "Dana": "A"}
    students = {
        "Anna": {"Jongen/meisje": "Meisje", "Jaarlaag": 6},
        "Bram": {"Jongen/meisje": "Jongen", "Jaarlaag": 6},
        "Cas": {"Jongen/meisje": "Jongen", "Jaarlaag": 7},
        "Dana": {"Jongen/meisje": "Meisje", "Jaarlaag": 7},
    }
    groups_to = {"A": {"Jongens": 0, "Meisjes": 0}}

    result = to_solution_result(
        _solution(assignment), _preferences(list(assignment)), students, groups_to
    )

    comp = result.group_composition["A"]
    assert comp.per_year == {
        6: SexCounts(boys=1, girls=1),
        7: SexCounts(boys=1, girls=1),
    }
    assert comp.boys_total == 2
    assert comp.girls_total == 2


def test_per_year_none_cohort_for_students_without_jaarlaag():
    """Students without a Jaarlaag (Excel input path) fall into the None cohort."""
    assignment = {"Anna": "A", "Bram": "A"}
    students = {
        "Anna": {"Jongen/meisje": "Meisje"},
        "Bram": {"Jongen/meisje": "Jongen"},
    }
    groups_to = {"A": {"Jongens": 1, "Meisjes": 0}}

    result = to_solution_result(
        _solution(assignment), _preferences(list(assignment)), students, groups_to
    )

    comp = result.group_composition["A"]
    assert comp.per_year == {None: SexCounts(boys=1, girls=1)}
    assert comp.boys_total == 2  # 1 current occupant + 1 newly assigned
    assert comp.girls_total == 1


def test_per_year_mixed_none_and_numbered_cohorts():
    """A group can mix a None cohort (hand-added student) with numbered ones."""
    assignment = {"Anna": "A", "Bram": "A"}
    students = {
        "Anna": {"Jongen/meisje": "Meisje", "Jaarlaag": 6},
        "Bram": {"Jongen/meisje": "Jongen"},
    }
    groups_to = {"A": {"Jongens": 0, "Meisjes": 0}}

    result = to_solution_result(
        _solution(assignment), _preferences(list(assignment)), students, groups_to
    )

    comp = result.group_composition["A"]
    assert comp.per_year == {
        6: SexCounts(boys=0, girls=1),
        None: SexCounts(boys=1, girls=0),
    }
