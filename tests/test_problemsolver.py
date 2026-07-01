"""Unit tests for ProblemSolver internal helpers."""

import math

import pandas as pd

from aliexpress.solver._balance import GroupBalance
from aliexpress.solver.problemsolver import ProblemSolver


def _empty_prefs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Waarde", "Gewicht"],
        index=pd.MultiIndex.from_tuples([], names=["Leerling", "TypeWens", "Nr"]),
    )


def _make_solver(
    students: dict, groups_to: dict = None, groupbalance: GroupBalance = None
) -> ProblemSolver:
    """Minimal ProblemSolver with no preferences."""
    if groups_to is None:
        groups_to = {
            "blauw": {"Jongens": 0, "Meisjes": 0},
            "rood": {"Jongens": 0, "Meisjes": 0},
        }
    kwargs = {}
    if groupbalance is not None:
        kwargs["groupbalance"] = groupbalance
    return ProblemSolver(_empty_prefs(), students, groups_to, [], **kwargs)


def _make_graag_met_prefs(
    pairs: list[tuple[str, str]], weight: float = 3.0
) -> pd.DataFrame:
    """Build a 'Graag met' preferences frame from (student, target) pairs."""
    records = []
    for nr, (student, target) in enumerate(pairs, start=1):
        records.append(
            {
                "Leerling": student,
                "TypeWens": "Graag met",
                "Nr": nr,
                "Waarde": target,
                "Gewicht": weight,
            }
        )
    if not records:
        return pd.DataFrame(
            columns=["Waarde", "Gewicht"],
            index=pd.MultiIndex.from_tuples([], names=["Leerling", "TypeWens", "Nr"]),
        )
    df = pd.DataFrame(records).set_index(["Leerling", "TypeWens", "Nr"])
    df.columns.name = "TypeWaarde"
    return df


# ---------------------------------------------------------------------------
# cohorts()
# ---------------------------------------------------------------------------


def test_cohorts_no_year_gives_single_none_cohort():
    """Students without Jaarlaag fall into one implicit cohort keyed by None."""
    students = {
        "anna": {
            "Stamgroep": "A",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
        },
        "bram": {
            "Stamgroep": "A",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
        },
    }
    solver = _make_solver(students)
    cohorts = solver.cohorts()
    assert list(cohorts.keys()) == [None]
    assert set(cohorts[None]) == {"anna", "bram"}


def test_cohorts_single_year_gives_one_cohort():
    """All students in the same Jaarlaag → one cohort (the doorzetten case)."""
    students = {
        "anna": {
            "Stamgroep": "A",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        },
        "bram": {
            "Stamgroep": "A",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        },
    }
    solver = _make_solver(students)
    cohorts = solver.cohorts()
    assert list(cohorts.keys()) == [6]
    assert set(cohorts[6]) == {"anna", "bram"}


def test_cohorts_multiple_years_split_correctly():
    """Multiple Jaarlaag values produce one cohort per distinct value."""
    students = {
        "anna": {
            "Stamgroep": "A",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        },
        "bram": {
            "Stamgroep": "A",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        },
        "cas": {
            "Stamgroep": "B",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 7,
        },
        "demi": {
            "Stamgroep": "B",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 7,
        },
        "eva": {
            "Stamgroep": "C",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 8,
        },
    }
    solver = _make_solver(students)
    cohorts = solver.cohorts()
    assert set(cohorts.keys()) == {6, 7, 8}
    assert set(cohorts[6]) == {"anna", "bram"}
    assert set(cohorts[7]) == {"cas", "demi"}
    assert set(cohorts[8]) == {"eva"}


def test_cohorts_mixed_present_absent_year():
    """Students with and without Jaarlaag are separated into distinct cohorts."""
    students = {
        "anna": {
            "Stamgroep": "A",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        },
        "bram": {
            "Stamgroep": "A",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
        },
    }
    solver = _make_solver(students)
    cohorts = solver.cohorts()
    assert set(cohorts.keys()) == {6, None}
    assert cohorts[6] == ["anna"]
    assert cohorts[None] == ["bram"]


# ---------------------------------------------------------------------------
# _constraint_equal_new_students per year (Jaarlaag)
# ---------------------------------------------------------------------------


def test_equal_new_students_per_year_enforced():
    """Per-year count constraint prevents year-6 students from clustering in one group.

    Without per-year constraints the solver can put all 4 year-6 students in 'blauw'
    (satisfying every cross-pair preference) while still satisfying the total-student
    balance (4 per group).  The per-year constraint (max_diff=0) forces exactly 2
    year-6 students per group, so the solver cannot cluster them.

    The preferences create a strong incentive for year-6 to cluster; the test asserts
    that the per-year constraint overrides that incentive.
    """

    # 4 year-6 (2M 2F) + 4 year-7 (2M 2F) → 2 groups of 4.
    def _s(grp, sex, year):
        return {
            "Stamgroep": grp,
            "Jongen/meisje": sex,
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": year,
        }

    students = {
        "a6": _s("A", "Jongen", 6),
        "b6": _s("A", "Meisje", 6),
        "c6": _s("B", "Jongen", 6),
        "d6": _s("B", "Meisje", 6),
        "a7": _s("C", "Jongen", 7),
        "b7": _s("C", "Meisje", 7),
        "c7": _s("D", "Jongen", 7),
        "d7": _s("D", "Meisje", 7),
    }
    # Strong mutual preferences within year-6 → solver wants to cluster them.
    prefs = _make_graag_met_prefs(
        [
            ("a6", "b6"),
            ("a6", "c6"),
            ("a6", "d6"),
            ("b6", "c6"),
            ("b6", "d6"),
            ("c6", "d6"),
        ],
        weight=3.0,
    )
    groups_to = {
        "blauw": {"Jongens": 0, "Meisjes": 0},
        "rood": {"Jongens": 0, "Meisjes": 0},
    }
    balance = GroupBalance(
        max_diff_n_students_year=0,
        max_diff_n_students_total=0,
        max_clique=4,
        max_clique_sex=2,
        max_imbalance_boys_girls_year=2,
        max_imbalance_boys_girls_total=2,
    )
    solver = ProblemSolver(prefs, students, groups_to, [], groupbalance=balance)
    solver.run()
    result = solver.extract_solution()

    for year in (6, 7):
        cohort = [s for s, info in students.items() if info.get("Jaarlaag") == year]
        # Include 0 for groups that received no students from this cohort.
        counts = {
            g: sum(1 for s in cohort if result.assignment[s] == g) for g in groups_to
        }
        diff = max(counts.values()) - min(counts.values())
        assert diff == 0, f"year {year}: counts={counts}, diff={diff}"
