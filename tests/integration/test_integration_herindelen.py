"""Integration test for herindelen: multi-year redistribution over empty groups.

Tests the solver in a realistic herindelen configuration:
- Three year groups (6, 7, 8), each with boys and girls in three stamgroepen.
- Three destination groups, all with occupancy=0 (no fixed students).
- All constraint types active: positive preferences, negative preferences,
  MinimaleTevredenheid, Niet-in (group exclusion), and Niet-samen rules.
- Fixed GroupBalance (known limits) so assertions are deterministic.
"""

import math

import pandas as pd

from aliexpress.solver._balance import GroupBalance
from aliexpress.solver.problemsolver import ProblemSolver

_GROUPS_TO = {
    "blauw": {"Jongens": 0, "Meisjes": 0},
    "rood": {"Jongens": 0, "Meisjes": 0},
    "geel": {"Jongens": 0, "Meisjes": 0},
}

# Fixed balance: generous enough to be feasible, tight enough to be asserted.
# max_clique=2: two from one stamgroep may land in the same group (b6_0 and g6_0
# share G6A; max_clique=2 allows them together).
# max_diff=0: each group receives exactly 2 students per year (6/3).
# max_imbalance_year=0: with 3B+3G per year over 3 groups this means 1B+1G per
#     group per year — already fully implied by max_diff=0 + equal gender split.
_BALANCE = GroupBalance(
    max_clique=2,
    max_clique_sex=2,
    max_diff_n_students_year=0,
    max_diff_n_students_total=0,
    max_imbalance_boys_girls_year=0,
    max_imbalance_boys_girls_total=0,
)


def _build_students():
    """18 students: years 6, 7, 8 × 3 stamgroepen × 1 boy + 1 girl.

    b6_0 has MinimaleTevredenheid=0.01.  All others have math.nan (no floor).
    """
    students = {}
    for year in (6, 7, 8):
        for i in range(3):
            grp = f"G{year}{chr(ord('A') + i)}"
            min_sat = 0.01 if (year == 6 and i == 0) else math.nan
            students[f"b{year}_{i}"] = {
                "Stamgroep": grp,
                "Jongen/meisje": "Jongen",
                "MinimaleTevredenheid": min_sat,
                "Jaarlaag": year,
            }
            students[f"g{year}_{i}"] = {
                "Stamgroep": grp,
                "Jongen/meisje": "Meisje",
                "MinimaleTevredenheid": math.nan,
                "Jaarlaag": year,
            }
    return students


def _build_prefs() -> pd.DataFrame:
    """Preferences for the herindelen scenario.

    - Positive ("Graag met"): b6_0 wants g6_0 (same stamgroep G6A, opposite gender).
      Both can land in the same group with max_clique=2 and 1B+1G per year per group.
    - Negative ("Liever niet met", weight already negated): g7_0 doesn't want g7_1.
    - Niet in: b8_0 may not go to "blauw".
    """
    records = [
        {
            "Leerling": "b6_0",
            "TypeWens": "Graag met",
            "Nr": 1,
            "Waarde": "g6_0",
            "Gewicht": 1.0,
        },
        {
            "Leerling": "g7_0",
            "TypeWens": "Liever niet met",
            "Nr": 1,
            "Waarde": "g7_1",
            "Gewicht": -1.0,
        },
        {
            "Leerling": "b8_0",
            "TypeWens": "Niet in",
            "Nr": 1,
            "Waarde": "blauw",
            "Gewicht": 1.0,
        },
    ]
    df = pd.DataFrame(records).set_index(["Leerling", "TypeWens", "Nr"])
    df.columns.name = "TypeWaarde"
    return df


def test_herindelen_multi_year_all_constraints():
    """Full herindelen solve: 18 students across 3 year groups, occupancy=0.

    All constraint types active:
    - Positive preference (b6_0 wants g6_0, same stamgroep, opposite gender).
    - Negative preference (g7_0 doesn't want g7_1).
    - MinimaleTevredenheid: b6_0 must reach at least 0.01 satisfaction.
    - Niet-in: b8_0 may not go to "blauw".
    - Niet-samen: at most 2 of {b6_0, b7_0, b8_0} in any single group.

    Structural assertions (fixed GroupBalance, so limits are known):
    - All 18 students are assigned.
    - b8_0 is not in "blauw" (Niet-in hard constraint).
    - No group contains more than 2 of {b6_0, b7_0, b8_0} (Niet-samen).
    - b6_0 achieves satisfaction >= 0.01 (MinimaleTevredenheid).
    - Per-year student count is exactly 2 per group (max_diff_n_students_year=0).
    - Per-year gender imbalance is 0 per group (max_imbalance_boys_girls_year=0).
    """
    students = _build_students()
    prefs = _build_prefs()
    # {b6_0, b7_0, b8_0}: one boy from each year; with 1B per year per group they would
    # all land in the same group if unconstrained.  Max=2 forces them apart.
    not_together = [{"group": {"b6_0", "b7_0", "b8_0"}, "Max_aantal_samen": 2}]

    solver = ProblemSolver(
        prefs, students, _GROUPS_TO, not_together, groupbalance=_BALANCE
    )
    solver.run()
    result = solver.extract_solution()

    assert set(result.assignment.keys()) == set(
        students.keys()
    ), "Not all students assigned"

    # Niet-in hard constraint.
    assert (
        result.assignment["b8_0"] != "blauw"
    ), "b8_0 ended up in blauw (Niet-in violated)"

    # Niet-samen hard constraint: at most 2 of the three boys in any one group.
    niet_samen = {"b6_0", "b7_0", "b8_0"}
    for grp in _GROUPS_TO:
        count = sum(1 for s in niet_samen if result.assignment[s] == grp)
        assert count <= 2, f"Niet-samen violated: {count} of {niet_samen} in {grp}"

    # MinimaleTevredenheid hard constraint.
    b6_0_sat = result.student_satisfaction["b6_0"]
    assert (
        b6_0_sat >= 0.01
    ), f"b6_0 satisfaction {b6_0_sat} < 0.01 (MinimaleTevredenheid)"

    # Per-year structural balance (asserted against the known fixed GroupBalance).
    for year in (6, 7, 8):
        cohort = [s for s, info in students.items() if info.get("Jaarlaag") == year]
        counts = {
            g: sum(1 for s in cohort if result.assignment[s] == g) for g in _GROUPS_TO
        }
        diff = max(counts.values()) - min(counts.values())
        assert (
            diff <= _BALANCE.max_diff_n_students_year
        ), f"year {year}: count diff={diff} > limit {_BALANCE.max_diff_n_students_year}"
        for grp in _GROUPS_TO:
            boys = sum(
                1
                for s in cohort
                if result.assignment[s] == grp
                and students[s]["Jongen/meisje"] == "Jongen"
            )
            girls = sum(
                1
                for s in cohort
                if result.assignment[s] == grp
                and students[s]["Jongen/meisje"] == "Meisje"
            )
            assert abs(boys - girls) <= _BALANCE.max_imbalance_boys_girls_year, (
                f"year {year}, group {grp}: gender imbalance |{boys}-{girls}|"
                f" > {_BALANCE.max_imbalance_boys_girls_year}"
            )


def test_herindelen_single_year_matches_doorzetten_behavior():
    """Herindelen with a single Jaarlaag behaves identically to the doorzetten path.

    When all students have the same Jaarlaag, cohorts() returns one cohort keyed by
    that year.  The per-year constraints degenerate to the same constraints as the
    old single-cohort code path, so the solver must produce a valid solution.
    """
    students = {}
    for i in range(3):
        students[f"b_{i}"] = {
            "Stamgroep": "A",
            "Jongen/meisje": "Jongen",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        }
        students[f"g_{i}"] = {
            "Stamgroep": "B",
            "Jongen/meisje": "Meisje",
            "MinimaleTevredenheid": math.nan,
            "Jaarlaag": 6,
        }

    groups_to = {
        "blauw": {"Jongens": 0, "Meisjes": 0},
        "rood": {"Jongens": 0, "Meisjes": 0},
        "geel": {"Jongens": 0, "Meisjes": 0},
    }
    balance = GroupBalance(
        max_clique=2,
        max_clique_sex=2,
        max_diff_n_students_year=1,
        max_diff_n_students_total=1,
        max_imbalance_boys_girls_year=2,
        max_imbalance_boys_girls_total=2,
    )
    prefs = pd.DataFrame(
        columns=["Waarde", "Gewicht"],
        index=pd.MultiIndex.from_tuples([], names=["Leerling", "TypeWens", "Nr"]),
    )
    solver = ProblemSolver(prefs, students, groups_to, [], groupbalance=balance)
    solver.run()
    result = solver.extract_solution()

    assert set(result.assignment.keys()) == set(students.keys())
    cohort = list(students.keys())
    counts = {g: sum(1 for s in cohort if result.assignment[s] == g) for g in groups_to}
    diff = max(counts.values()) - min(counts.values())
    assert diff <= balance.max_diff_n_students_year
