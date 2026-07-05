"""Integration test for herindelen: multi-year redistribution over empty groups.

Tests the solver in a realistic herindelen configuration:
- Three year groups (6, 7, 8), each with boys and girls in three stamgroepen.
- Three destination groups, all with occupancy=0 (no fixed students).
- All constraint types active: positive preferences, negative preferences,
  MinimaleTevredenheid, Niet-in (group exclusion), and Niet-samen rules.
- Fixed GroupBalance (known limits) so assertions are deterministic.
"""

import math
import time

import pandas as pd

from aliexpress.solver._balance import GroupBalance
from aliexpress.solver.cpsat import engine
from aliexpress.solver.cpsat.results import to_solution_result

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
    - Negative: g7_0 doesn't want g7_1.  The solver sees preferences *post-negation*
      (datareader renames "Liever niet met" to "Graag met" with negated weight), so
      that is the convention used here.
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
            "TypeWens": "Graag met",
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

    solution = engine.solve_with_fixed_balance(
        preferences=prefs,
        students=students,
        groups_to=_GROUPS_TO,
        not_together=not_together,
        groupbalance=_BALANCE,
    )
    result = to_solution_result(solution, prefs, students, _GROUPS_TO)

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
    solution = engine.solve_with_fixed_balance(
        preferences=prefs,
        students=students,
        groups_to=groups_to,
        not_together=[],
        groupbalance=balance,
    )
    result = to_solution_result(solution, prefs, students, groups_to)

    assert set(result.assignment.keys()) == set(students.keys())
    cohort = list(students.keys())
    counts = {g: sum(1 for s in cohort if result.assignment[s] == g) for g in groups_to}
    diff = max(counts.values()) - min(counts.values())
    assert diff <= balance.max_diff_n_students_year


# ── Realistic-scale test ──────────────────────────────────────────────────────

_GROUPS_REALISTIC = {
    "blauw": {"Jongens": 0, "Meisjes": 0},
    "rood": {"Jongens": 0, "Meisjes": 0},
    "geel": {"Jongens": 0, "Meisjes": 0},
    "groen": {"Jongens": 0, "Meisjes": 0},
}

# Year 6: 28 students / 4 groups = 7 per group.
# Year 7: 32 / 4 = 8.  Year 8: 28 / 4 = 7.
# Gender per year: 14B+14G for years 6,8 → 3.5B per group → max_imbalance_year=1.
# 16B+16G for year 7 → 4B per group exactly → covered by max_imbalance_year=1.
# Stamgroepen of 14-16 students over 4 groups → max 4 per stamgroep per group.
_BALANCE_REALISTIC = GroupBalance(
    max_clique=4,
    max_clique_sex=2,
    max_diff_n_students_year=1,
    max_diff_n_students_total=1,
    max_imbalance_boys_girls_year=1,
    max_imbalance_boys_girls_total=1,
)

_NOT_TOGETHER_REALISTIC = [
    {"group": {"j6A_0", "j6B_0", "j7A_0"}, "Max_aantal_samen": 2},
    {"group": {"m7A_0", "m7B_0", "m8A_0"}, "Max_aantal_samen": 2},
    {"group": {"j8A_0", "j8B_0", "m8A_0"}, "Max_aantal_samen": 1},
]


def _build_realistic_students() -> dict:
    """88 students: 3 year groups × 2 stamgroepen × equal boys/girls.

    Year 6: 2 × (7B + 7G) = 28  (stamgroepen 6A, 6B)
    Year 7: 2 × (8B + 8G) = 32  (stamgroepen 7A, 7B)
    Year 8: 2 × (7B + 7G) = 28  (stamgroepen 8A, 8B)
    Total: 88 → 4 destination groups × 22.

    Students at index 0 of each stamgroep get MinimaleTevredenheid=0.01 (~7%).
    """
    students = {}
    config = [
        (6, "A", 7),
        (6, "B", 7),
        (7, "A", 8),
        (7, "B", 8),
        (8, "A", 7),
        (8, "B", 7),
    ]
    for year, grp_letter, per_gender in config:
        stamgroep = f"{year}{grp_letter}"
        for i in range(per_gender):
            min_sat = 0.01 if i == 0 else math.nan
            students[f"j{stamgroep}_{i}"] = {
                "Stamgroep": stamgroep,
                "Jongen/meisje": "Jongen",
                "MinimaleTevredenheid": min_sat,
                "Jaarlaag": year,
            }
            students[f"m{stamgroep}_{i}"] = {
                "Stamgroep": stamgroep,
                "Jongen/meisje": "Meisje",
                "MinimaleTevredenheid": math.nan,
                "Jaarlaag": year,
            }
    return students


def _build_realistic_prefs(students: dict) -> pd.DataFrame:
    """Deterministic realistic preferences for 88 students.

    Each student gets 1–5 positive wishes (Graag met) targeting nearby students
    in the sorted key list.  ~20% also have a negative wish (post-negation
    convention: "Graag met" with negative weight, as the datareader hands them
    to the solver) targeting a student a third of the list ahead.  ~12% have a
    Niet-in restriction for one of the four destination groups.  No
    self-preferences; no duplicate (Leerling, TypeWens, Nr) index entries.
    """
    group_names = list(_GROUPS_REALISTIC.keys())
    keys = sorted(students.keys())
    n = len(keys)
    records = []

    for idx, leerling in enumerate(keys):
        n_pos = (idx % 5) + 1
        for k in range(n_pos):
            records.append(
                {
                    "Leerling": leerling,
                    "TypeWens": "Graag met",
                    "Nr": k + 1,
                    "Waarde": keys[(idx + k + 1) % n],
                    "Gewicht": 1.0,
                }
            )

        if idx % 5 == 2:
            records.append(
                {
                    "Leerling": leerling,
                    "TypeWens": "Graag met",
                    "Nr": n_pos + 1,
                    "Waarde": keys[(idx + n // 3) % n],
                    "Gewicht": -1.0,
                }
            )

        if idx % 8 == 0:
            records.append(
                {
                    "Leerling": leerling,
                    "TypeWens": "Niet in",
                    "Nr": 1,
                    "Waarde": group_names[idx % len(group_names)],
                    "Gewicht": 1.0,
                }
            )

    df = pd.DataFrame(records).set_index(["Leerling", "TypeWens", "Nr"])
    df.columns.name = "TypeWaarde"
    return df


#: Generous upper bound on wall time. This instance's specific Niet-samen
#: student selection happens to be a hard CP-SAT proof (see
#: benchmarks/spike_cpsat.py for a same-shape instance that proves in seconds
#: when the Niet-samen students are drawn from a single stamgroep instead of
#: spread across cohorts), and repeated measurements of this exact instance
#: ranged 347-843s: CP-SAT's num_workers=8 races several search strategies in
#: parallel threads, so a fixed random_seed makes the proof itself
#: deterministic but not the wall-clock time to reach it (which thread finds
#: and propagates a good bound first depends on real-time OS scheduling).
#: 1800s gives ample headroom over the observed spread. Still dramatically
#: better than the old pulp/HiGHS backend, which never finished this instance.
_REALISTIC_TIME_LIMIT_S = 1800


def _assert_minimale_tevredenheid(result, students: dict) -> None:
    """Every student with a MinimaleTevredenheid floor reaches at least it."""
    for student, info in students.items():
        floor = info["MinimaleTevredenheid"]
        if math.isnan(floor):
            continue
        sat = result.student_satisfaction[student]
        assert sat >= floor, f"{student} satisfaction {sat} < floor {floor}"


def _assert_niet_samen(result, groups_to: dict) -> None:
    """Every Niet-samen rule stays within its Max_aantal_samen, per group."""
    for rule in _NOT_TOGETHER_REALISTIC:
        for grp in groups_to:
            count = sum(1 for s in rule["group"] if result.assignment[s] == grp)
            assert (
                count <= rule["Max_aantal_samen"]
            ), f"Niet-samen violated: {count} of {rule['group']} in {grp}"


def _gender_counts(
    result, students: dict, groups_to: dict, members
) -> dict[str, tuple[int, int]]:
    """Per group, the (boys, girls) count among ``members`` in the assignment."""
    counts = {}
    for grp in groups_to:
        boys = sum(
            1
            for s in members
            if students[s]["Jongen/meisje"] == "Jongen" and result.assignment[s] == grp
        )
        girls = sum(
            1
            for s in members
            if students[s]["Jongen/meisje"] == "Meisje" and result.assignment[s] == grp
        )
        counts[grp] = (boys, girls)
    return counts


def _assert_year_balance(result, students: dict, groups_to: dict) -> None:
    """Per-year student-count spread and gender imbalance stay within the fixed balance."""
    for year in (6, 7, 8):
        cohort = [s for s, info in students.items() if info.get("Jaarlaag") == year]
        counts = {
            g: sum(1 for s in cohort if result.assignment[s] == g) for g in groups_to
        }
        diff = max(counts.values()) - min(counts.values())
        assert diff <= _BALANCE_REALISTIC.max_diff_n_students_year, (
            f"year {year}: count diff={diff} > limit "
            f"{_BALANCE_REALISTIC.max_diff_n_students_year}"
        )
        for grp, (boys, girls) in _gender_counts(
            result, students, groups_to, cohort
        ).items():
            assert (
                abs(boys - girls) <= _BALANCE_REALISTIC.max_imbalance_boys_girls_year
            ), (
                f"year {year}, group {grp}: gender imbalance |{boys}-{girls}|"
                f" > {_BALANCE_REALISTIC.max_imbalance_boys_girls_year}"
            )


def _assert_total_balance(result, students: dict, groups_to: dict) -> None:
    """Whole-group student-count spread and gender imbalance stay within the fixed balance."""
    totals = {
        g: sum(1 for s in students if result.assignment[s] == g) for g in groups_to
    }
    total_diff = max(totals.values()) - min(totals.values())
    assert total_diff <= _BALANCE_REALISTIC.max_diff_n_students_total, (
        f"total count diff={total_diff} > limit "
        f"{_BALANCE_REALISTIC.max_diff_n_students_total}"
    )
    for grp, (boys, girls) in _gender_counts(
        result, students, groups_to, students
    ).items():
        assert abs(boys - girls) <= _BALANCE_REALISTIC.max_imbalance_boys_girls_total, (
            f"group {grp}: total gender imbalance |{boys}-{girls}|"
            f" > {_BALANCE_REALISTIC.max_imbalance_boys_girls_total}"
        )


def test_herindelen_realistic_scale():
    """Scale + timing baseline: 88 students (4 × 22), 3 year groups, realistic prefs.

    Realistic constraint mix: 1–5 positive wishes per student, ~20% negative,
    ~12% Niet-in, 3 Niet-samen rules, per-year gender and count balance.
    """
    students = _build_realistic_students()
    prefs = _build_realistic_prefs(students)

    t0 = time.perf_counter()
    solution = engine.solve_with_fixed_balance(
        preferences=prefs,
        students=students,
        groups_to=_GROUPS_REALISTIC,
        not_together=_NOT_TOGETHER_REALISTIC,
        groupbalance=_BALANCE_REALISTIC,
    )
    elapsed = time.perf_counter() - t0
    result = to_solution_result(solution, prefs, students, _GROUPS_REALISTIC)
    print(
        f"\nherindelen realistic-scale: {elapsed:.1f}s"
        f" ({len(students)} students, {len(_GROUPS_REALISTIC)} groups)"
    )

    assert set(result.assignment.keys()) == set(
        students.keys()
    ), "Not all students assigned"
    assert (
        elapsed <= _REALISTIC_TIME_LIMIT_S
    ), f"Solve took {elapsed:.1f}s, exceeding the {_REALISTIC_TIME_LIMIT_S}s budget"

    _assert_minimale_tevredenheid(result, students)
    _assert_niet_samen(result, _GROUPS_REALISTIC)
    _assert_year_balance(result, students, _GROUPS_REALISTIC)
    _assert_total_balance(result, students, _GROUPS_REALISTIC)
