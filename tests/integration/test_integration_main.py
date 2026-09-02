"""Integration tests for the main functionality of the AliExpress application.

These tests ensure the student distribution process works on both a small and a full
dataset. The exact group *labels* of an assignment are a degenerate optimum (the groups
can be relabelled, e.g. "Blauw" <-> "Geel", without changing who sits with whom), so the
group-placement tables are checked on their structure and respected class balance rather
than cell-by-cell. The per-student satisfaction - the actual optimization objective - is
uniquely determined and is asserted in full."""

import json
from collections import Counter

import pandas as pd
import pytest

from aliexpress import errors
from aliexpress.data import datareader
from aliexpress.data.datareader import GroupCounts
from aliexpress.data.preferences_data import PreferenceData
from aliexpress.data.preferences_form import (
    Preference,
    PreferenceKind,
    StudentEntry,
    build_preference_data,
)
from aliexpress.main import distribute_students_from_data, distribute_students_once
from aliexpress.solver import engine
from aliexpress.solver._balance import BalanceMaxima, GroupBalance
from aliexpress.solver._balance_families import SLACK_WEIGHTS, STRICTEST_LIMIT
from aliexpress.solver.groepsindeling_view import GroepsindelingView

_NOT_TOGETHER_SMALL = [
    {"group": {"Claire", "Bram", "Eva", "Daan"}, "Max_aantal_samen": 2},
]

_NOT_TOGETHER_FULL = [
    {"group": {"Daan", "Anne"}, "Max_aantal_samen": 1},
    {"group": {"Noor", "Naomi"}, "Max_aantal_samen": 1},
    {
        "group": {"Stijn", "Cas", "David", "Adam", "Julian", "Jurre", "Tijn", "Jayden"},
        "Max_aantal_samen": 3,
    },
]


_EXPECTED_KEYS = {
    "Overgangsmatrix",
    "Leerlingtevredenheid",
    "VervuldeVoorkeuren",
}

_SMALL_SATISFACTION = {
    "Tevredenheid": {
        "Anna": 0.903226,
        "Bram": 1.0,
        "Claire": 0.944882,
        "Daan": 0.937614,
        "Eva": 0.0,
    },
    "Aantal gehonoreerde voorkeuren": {
        "Anna": 3.0,
        "Bram": 3.0,
        "Claire": 4.0,
        "Daan": 4.0,
        "Eva": 0.0,
    },
    "Aantal voorkeuren": {
        "Anna": 5.0,
        "Bram": 3.0,
        "Claire": 7.0,
        "Daan": 13.0,
        "Eva": 2.0,
    },
}

_FULL_SATISFACTION = {
    "Tevredenheid": {
        "Adam": 0.516129,
        "Amy": 0.6673,
        "Anna": 0.976378,
        "Anne": 0.607369,
        "Anne Claire": 0.571429,
        "Benjamin": 1.0,
        "Bram": 0.774194,
        "Cas": 0.857143,
        "Daan": 1.0,
        "David": 0.857143,
        "Eline": 0.661054,
        "Emily": 0.976378,
        "Esmee": 0.607369,
        "Feline": 0.571429,
        "Fenna": 0.784678,
        "Iris": 0.571429,
        "Jack": 0.861287,
        "Jayden": 0.523119,
        "Jill": 0.666667,
        "Julia": 0.984127,
        "Julian": 0.709125,
        "Jurre": 0.666667,
        "Lars": 1.0,
        "Liv A.": 0.661054,
        "Liv B.": 0.656708,
        "Lois": 1.0,
        "Lotte": 0.784678,
        "Lucas": 0.571429,
        "Lynn": 1.0,
        "Mats": 0.651537,
        "Max": 0.661054,
        "Naomi": 1.0,
        "Nina": 0.8,
        "Noor": 0.571429,
        "Nora": 0.933333,
        "Siem X.": 0.666667,
        "Siem Y.": 0.822719,
        "Sophie": 0.666667,
        "Stijn": 0.533333,
        "Sven": 1.0,
        "Tijn": 0.666667,
        "Vera": 0.533333,
        "Zoe": 0.979573,
    },
    "Aantal gehonoreerde voorkeuren": {
        "Adam": 1.0,
        "Amy": 1.5,
        "Anna": 5.0,
        "Anne": 1.0,
        "Anne Claire": 1.0,
        "Benjamin": 3.0,
        "Bram": 2.0,
        "Cas": 2.0,
        "Daan": 2.0,
        "David": 2.0,
        "Eline": 1.5,
        "Emily": 5.0,
        "Esmee": 1.0,
        "Feline": 1.0,
        "Fenna": 2.0,
        "Iris": 1.0,
        "Jack": 2.5,
        "Jayden": 1.0,
        "Jill": 1.0,
        "Julia": 5.0,
        "Julian": 1.5,
        "Jurre": 1.0,
        "Lars": 2.0,
        "Liv A.": 1.5,
        "Liv B.": 1.5,
        "Lois": 1.0,
        "Lotte": 2.0,
        "Lucas": 1.0,
        "Lynn": 1.0,
        "Mats": 1.5,
        "Max": 1.5,
        "Naomi": 1.0,
        "Nina": 2.0,
        "Noor": 1.0,
        "Nora": 3.0,
        "Siem X.": 1.0,
        "Siem Y.": 2.0,
        "Sophie": 1.0,
        "Stijn": 1.0,
        "Sven": 2.0,
        "Tijn": 1.0,
        "Vera": 1.0,
        "Zoe": 5.0,
    },
    "Aantal voorkeuren": {
        "Adam": 5.0,
        "Amy": 5.0,
        "Anna": 7.0,
        "Anne": 2.5,
        "Anne Claire": 3.0,
        "Benjamin": 3.0,
        "Bram": 5.0,
        "Cas": 3.0,
        "Daan": 2.0,
        "David": 3.0,
        "Eline": 5.5,
        "Emily": 7.0,
        "Esmee": 2.5,
        "Feline": 3.0,
        "Fenna": 4.5,
        "Iris": 3.0,
        "Jack": 4.5,
        "Jayden": 4.5,
        "Jill": 2.0,
        "Julia": 6.0,
        "Julian": 3.5,
        "Jurre": 2.0,
        "Lars": 2.0,
        "Liv A.": 5.5,
        "Liv B.": 6.0,
        "Lois": 1.0,
        "Lotte": 4.5,
        "Lucas": 3.0,
        "Lynn": 1.0,
        "Mats": 7.0,
        "Max": 5.5,
        "Naomi": 1.0,
        "Nina": 4.0,
        "Noor": 3.0,
        "Nora": 4.0,
        "Siem X.": 2.0,
        "Siem Y.": 3.5,
        "Sophie": 2.0,
        "Stijn": 4.0,
        "Sven": 2.0,
        "Tijn": 2.0,
        "Vera": 4.0,
        "Zoe": 6.5,
    },
}


def _tables(result):
    """Assert the result shape and return the three analysis tables plus the view."""
    assert isinstance(result, dict)
    assert "download" in result
    pd.read_excel(result["download"])  # download must be a readable Excel file

    assert "dataframes" in result
    dfs = result["dataframes"]
    assert isinstance(dfs, dict)
    assert _EXPECTED_KEYS.issubset(dfs.keys())

    assert isinstance(dfs["Overgangsmatrix"], pd.DataFrame)
    assert isinstance(dfs["Leerlingtevredenheid"], pd.io.formats.style.Styler)
    assert isinstance(dfs["VervuldeVoorkeuren"], pd.io.formats.style.Styler)
    assert isinstance(result["groepsindeling_view"], GroepsindelingView)
    return dfs, result["groepsindeling_view"]


def _assert_consistency(dfs, view, groups, stamgroepen):
    """Structural + counting invariants that hold for any optimal assignment."""
    assert set(view.group_order) == groups
    assert {c.name for c in view.groups} == groups

    # Each balance-row cell splits into boys + girls that sum to its count.
    for row in view.balance_rows:
        for group in view.group_order:
            count, boys, girls = row.per_group[group]
            assert count == boys + girls

    trans = dfs["Overgangsmatrix"]
    assert trans.index.name == "Stamgroep"
    assert set(trans.columns) == groups
    assert set(trans.index) == stamgroepen

    tevr = dfs["Leerlingtevredenheid"].data
    voorkeuren = dfs["VervuldeVoorkeuren"].data
    # Every student appears once in the satisfaction tables and is placed exactly once.
    n_students = len(tevr)
    assert set(voorkeuren.index) == set(tevr.index)
    assert trans.to_numpy().sum() == n_students
    movers = sum(
        cnt
        for r in view.balance_rows
        if not r.is_total
        for (cnt, _, _) in r.per_group.values()
    )
    assert movers == n_students
    return tevr, view


def _assert_satisfaction(tevr, expected):
    """The full per-student satisfaction table is the uniquely determined objective."""
    expected_df = pd.DataFrame(expected)
    expected_df.index.name = "Leerling"
    pd.testing.assert_frame_equal(
        tevr.round(6).sort_index(), expected_df.sort_index(), check_like=True
    )


def test_distribute_students_once_happy_flow_small():
    """Small, quick dataset with a manual balance (override path)."""
    result = distribute_students_once(
        path_preferences="tests/integration/voorkeuren_small.xlsx",
        path_groups_to="tests/integration/groepen_small.xlsx",
        not_together=_NOT_TOGETHER_SMALL,
        groupbalance=GroupBalance(max_imbalance_boys_girls_total=7),
    )
    dfs, view = _tables(result)
    tevr, view = _assert_consistency(dfs, view, {"Beren", "Otters"}, {"A", "B", "D"})
    _assert_satisfaction(tevr, _SMALL_SATISFACTION)

    trans = dfs["Overgangsmatrix"]
    total_row = next(r for r in view.balance_rows if r.is_total)
    year_rows = [r for r in view.balance_rows if not r.is_total]
    # Manual balance: GroupBalance(max_imbalance_boys_girls_total=7) + loose defaults.
    assert trans.to_numpy().max() <= 5  # max_clique
    assert total_row.sex_imbalance <= 7  # max_imbalance_boys_girls_total
    assert max(r.sex_imbalance for r in year_rows) <= 2  # max_imbalance_boys_girls_year
    assert total_row.size_diff <= 3
    assert max(r.size_diff for r in year_rows) <= 2


def _max_clique_sex(view) -> int:
    """Largest same-Stamgroep, same-sex headcount placed together in one target group.

    Tallies each group card's student chips by ``(Stamgroep, sex)`` across its jaarlaag
    sections -- ``chip.origin_full`` is the Stamgroep -- and returns the largest count found
    in any group. This is the public-view-model equivalent of the ``clique_sex`` balance
    family (see ``_BalanceFamilies._cliques``), read from ``GroepsindelingView`` rather than
    solver internals.
    """
    largest = 0
    for group in view.groups:
        counts: Counter = Counter()
        for section in group.year_sections:
            for chip in section.boys.students:
                counts[(chip.origin_full, "Jongen")] += 1
            for chip in section.girls.students:
                counts[(chip.origin_full, "Meisje")] += 1
        if counts:
            largest = max(largest, *counts.values())
    return largest


def _sorted_weighted_slacks(trans: pd.DataFrame, view) -> list[int]:
    """The realized weighted slacks, sorted large-to-small, one entry per family.

    Mirrors the quantity the balance stage's leximin actually optimizes (ADR-0018): each
    family's realized value becomes a slack via ``max(0, realized - STRICTEST_LIMIT)``,
    weighted by ``SLACK_WEIGHTS``, and the six weighted values are sorted descending. Every
    realized value is read from public result data: ``trans`` (the Overgangsmatrix) for
    ``clique``, ``view.balance_rows`` for the four size/gender families, and
    ``_max_clique_sex`` for ``clique_sex``.
    """
    total_row = next(r for r in view.balance_rows if r.is_total)
    year_rows = [r for r in view.balance_rows if not r.is_total]
    realized = {
        "diff_year": max(r.size_diff for r in year_rows),
        "diff_total": total_row.size_diff,
        "clique": int(trans.to_numpy().max()),
        "clique_sex": _max_clique_sex(view),
        "gender_year": max(r.sex_imbalance for r in year_rows),
        "gender_total": total_row.sex_imbalance,
    }
    weighted = [
        SLACK_WEIGHTS[name] * max(0, value - STRICTEST_LIMIT)
        for name, value in realized.items()
    ]
    return sorted(weighted, reverse=True)


def test_distribute_students_once_happy_flow_full():
    """Full dataset, satisfaction maximized within the auto-determined minimal balance
    relaxation that still lets every student fulfil at least one positive wish."""
    result = distribute_students_once(
        path_preferences="tests/integration/voorkeuren.xlsx",
        path_groups_to="tests/integration/groepen.xlsx",
        not_together=_NOT_TOGETHER_FULL,
    )
    dfs, view = _tables(result)
    tevr, view = _assert_consistency(
        dfs,
        view,
        {"Blauw", "Geel", "Groen", "Oranje"},
        {"Kaboutertuin", "Torteltuin", "Tovertuin", "Vlindertuin"},
    )
    _assert_satisfaction(tevr, _FULL_SATISFACTION)
    assert (tevr["Tevredenheid"] > 0).all()  # the goal: every student ends up positive

    trans = dfs["Overgangsmatrix"]
    # The balance stage leximin-minimizes the sorted weighted slacks across the six
    # balance families (ADR-0018): it spreads relaxation as evenly as the instance allows,
    # rather than piling it onto one family while leaving another slack. A ceiling on each
    # family separately would also accept a *stacked* vector (all the relaxation
    # concentrated in one or two families) that this instance's leximin optimum no longer
    # produces, so it is the sorted weighted slacks themselves that are pinned here.
    assert _sorted_weighted_slacks(trans, view) == [200, 200, 100, 100, 98, 0]


def _dataframe(table):
    """Return the underlying DataFrame for a result table (Styler or plain frame)."""
    return table.data if isinstance(table, pd.io.formats.style.Styler) else table


def test_distribute_students_from_json_matches_xlsx():
    """A run driven from a JSON-serialised PreferenceData equals the plain xlsx run.

    Reads the synthetic small example, builds a PreferenceData, serialises it to JSON and
    back, and asserts the resulting tables are identical to a normal distribute-from-xlsx
    run. This proves the JSON source path is lossless end to end (never real student data).
    """
    common = {
        "path_groups_to": "tests/integration/groepen_small.xlsx",
        "not_together": _NOT_TOGETHER_SMALL,
        "groupbalance": GroupBalance(max_imbalance_boys_girls_total=7),
    }

    from_xlsx = distribute_students_once(
        path_preferences="tests/integration/voorkeuren_small.xlsx", **common
    )

    target_groups = datareader.read_groups_excel(common["path_groups_to"])
    processor = datareader.VoorkeurenProcessor(
        "tests/integration/voorkeuren_small.xlsx"
    )
    processor.process(all_to_groups=list(target_groups.counts.keys()))
    restored = PreferenceData.from_json(processor.to_preference_data().to_json())

    from_json = distribute_students_from_data(
        restored,
        target_groups,
        not_together=common["not_together"],
        groupbalance=common["groupbalance"],
    )

    assert from_xlsx["dataframes"].keys() == from_json["dataframes"].keys()
    for key in from_xlsx["dataframes"]:
        pd.testing.assert_frame_equal(
            _dataframe(from_xlsx["dataframes"][key]),
            _dataframe(from_json["dataframes"][key]),
        )
    # The structured view is identical too: both paths have an empty unique_name map,
    # so the frozen dataclasses compare equal.
    assert from_xlsx["groepsindeling_view"] == from_json["groepsindeling_view"]


def _load_native_scenario(json_path, groups_path, not_together_path):
    """Load a native-format scenario: voorkeuren.json + groups.xlsx + not_together.json.

    This mirrors exactly what the wizard's start_distribution route does in production,
    so these helpers exercise the live code path rather than a test-only shortcut.
    """
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    preference_data = PreferenceData.from_json(json.dumps(payload))

    target_groups = datareader.read_groups_excel(groups_path)

    with open(not_together_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    not_together = [
        {"group": set(r["group"]), "Max_aantal_samen": r["Max_aantal_samen"]}
        for r in raw
    ]
    return preference_data, target_groups, not_together


def test_distribute_from_native_files_small():
    """Small scenario run from native files (voorkeuren_small.json + not_together_small.json).

    Proves the native-file path is lossless: same input data, same satisfaction table as
    the xlsx-driven test.
    """
    preference_data, target_groups, not_together = _load_native_scenario(
        "tests/integration/voorkeuren_small.json",
        "tests/integration/groepen_small.xlsx",
        "tests/integration/not_together_small.json",
    )

    result = distribute_students_from_data(
        preference_data,
        target_groups,
        not_together=not_together,
        groupbalance=GroupBalance(max_imbalance_boys_girls_total=7),
    )

    dfs, view = _tables(result)
    tevr, _ = _assert_consistency(dfs, view, {"Beren", "Otters"}, {"A", "B", "D"})
    _assert_satisfaction(tevr, _SMALL_SATISFACTION)


def test_distribute_from_native_files_full():
    """Full scenario run from native files (voorkeuren_full.json + not_together_full.json).

    Same assertion as test_distribute_students_once_happy_flow_full: the native path
    and the xlsx path must produce identical satisfaction values.
    """
    preference_data, target_groups, not_together = _load_native_scenario(
        "tests/integration/voorkeuren_full.json",
        "tests/integration/groepen.xlsx",
        "tests/integration/not_together_full.json",
    )

    result = distribute_students_from_data(
        preference_data,
        target_groups,
        not_together=not_together,
    )

    dfs, view = _tables(result)
    tevr, _ = _assert_consistency(
        dfs,
        view,
        {"Blauw", "Geel", "Groen", "Oranje"},
        {"Kaboutertuin", "Torteltuin", "Tovertuin", "Vlindertuin"},
    )
    _assert_satisfaction(tevr, _FULL_SATISFACTION)
    assert (tevr["Tevredenheid"] > 0).all()


def test_solver_stacks_duplicate_group_preferences():
    """The solver accepts duplicate group preferences and stacks them (ADR 0004).

    John names group 'Blauw' twice (a strong and a mild pull); Jane names it once. Both are
    placed in Blauw and fully satisfied, but John's wish total exceeds Jane's — proving the
    two group preferences are kept distinct and added, not collapsed into one.
    """

    def together(target, weight):
        return Preference(target=target, weight=weight, kind=PreferenceKind.TOGETHER)

    students = [
        StudentEntry(
            "John",
            "Jongen",
            "A",
            None,
            preferences=[together("Blauw", 2.0), together("Blauw", 0.5)],
        ),
        StudentEntry("Jane", "Meisje", "B", None, preferences=[together("Blauw", 2.0)]),
        StudentEntry("Tom", "Jongen", "C", None),
        StudentEntry("Sara", "Meisje", "D", None),
    ]
    target_groups = GroupCounts(
        counts={
            "blauw": {"Jongens": 0, "Meisjes": 0},
            "rood": {"Jongens": 0, "Meisjes": 0},
        },
        display={"blauw": "Blauw", "rood": "Rood"},
    )
    preference_data = build_preference_data(students, all_to_groups=["blauw", "rood"])

    result = distribute_students_from_data(preference_data, target_groups)

    dfs, _ = _tables(result)  # solver ran and produced the full, downloadable report
    tevr = dfs["Leerlingtevredenheid"].data
    assert tevr.loc["John", "Tevredenheid"] == 1.0
    assert (
        tevr.loc["John", "Aantal gehonoreerde voorkeuren"]
        == tevr.loc["John", "Aantal voorkeuren"]
    )
    # Stacking: John (two group preferences) outweighs Jane (one) — not collapsed.
    assert tevr.loc["John", "Aantal voorkeuren"] > tevr.loc["Jane", "Aantal voorkeuren"]


def test_distribute_students_once_happy_flow_infeasible():
    """Infeasible constraints must raise a FeasibilityError with a relaxation suggestion."""
    with pytest.raises(errors.FeasibilityError) as exc:
        distribute_students_once(
            path_preferences="tests/integration/voorkeuren.xlsx",
            path_groups_to="tests/integration/groepen.xlsx",
            not_together=_NOT_TOGETHER_FULL,
            groupbalance=GroupBalance(
                max_clique=1,
                max_clique_sex=1,
                max_diff_n_students_year=1,
                max_diff_n_students_total=1,
                max_imbalance_boys_girls_year=1,
                max_imbalance_boys_girls_total=1,
            ),
        )
    # A fixed class balance that admits no valid assignment gets one generic Dutch
    # message (no per-limit breakdown for CP-SAT yet: parked for a later balance
    # redesign, see CLAUDE.md).
    assert exc.value.code == "infeasible_problem"
    assert "ruimere klassenbalans" in exc.value.context["possible_improvement"]


def _one_stamgroep_scenario(n_students: int, groups: list[str]):
    """A tiny automatic-path scenario: ``n_students`` from one Stamgroep.

    Everyone shares Stamgroep "A", so the clique family binds. Each student has a
    single mild group preference (towards ``groups[0]``) so the report tables are
    non-empty; the preferences do not affect clique feasibility, which depends
    only on the Stamgroep.
    """
    preferences = [
        Preference(target=groups[0], weight=1.0, kind=PreferenceKind.TOGETHER)
    ]
    students = [
        StudentEntry(
            f"Kind{i}",
            "Jongen" if i % 2 else "Meisje",
            "A",
            None,
            preferences=preferences,
        )
        for i in range(n_students)
    ]
    keys = [g.lower() for g in groups]
    target_groups = GroupCounts(
        counts={key: {"Jongens": 0, "Meisjes": 0} for key in keys},
        display=dict(zip(keys, groups)),
    )
    preference_data = build_preference_data(students, all_to_groups=keys)
    return preference_data, target_groups


def test_empty_maxima_matches_no_maxima_on_automatic_path():
    """An empty BalanceMaxima() reproduces the automatic path's result exactly."""
    preference_data, target_groups = _one_stamgroep_scenario(
        5, ["Rood", "Geel", "Blauw"]
    )

    baseline = distribute_students_from_data(preference_data, target_groups)
    with_empty = distribute_students_from_data(
        preference_data, target_groups, maxima=BalanceMaxima()
    )

    pd.testing.assert_frame_equal(
        baseline["dataframes"]["Leerlingtevredenheid"].data,
        with_empty["dataframes"]["Leerlingtevredenheid"].data,
    )


def test_too_tight_cap_raises_actionable_feasibility_error():
    """A cap that alone makes the instance infeasible raises an actionable error.

    Five students from one Stamgroep over three groups need a clique of at least
    ceil(5/3) = 2 per group; capping ``max_clique`` at 1 is therefore impossible.
    Uncapped the same instance solves, so the cap is the sole cause — and the
    error must not misattribute it to the (empty) preferences.
    """
    preference_data, target_groups = _one_stamgroep_scenario(
        5, ["Rood", "Geel", "Blauw"]
    )

    # Uncapped: solves fine.
    distribute_students_from_data(
        preference_data, target_groups, maxima=BalanceMaxima(max_clique=None)
    )

    with pytest.raises(errors.FeasibilityError) as exc:
        distribute_students_from_data(
            preference_data, target_groups, maxima=BalanceMaxima(max_clique=1)
        )
    assert exc.value.code == "balance_caps_too_tight"
    assert exc.value.context == {
        "suggestion": {"clique": {"current": 1, "suggested": 2}}
    }


def _popular_student_scenario(n_students: int, groups: list[str]):
    """Everyone wants the one popular classmate; each from their own Stamgroep.

    ``Kind0`` is the popular student and has no preference of their own; every
    other student's single preference is "Graag met Kind0". The only way to
    honour all those preferences is to put everyone in Kind0's group, so the
    minimal relaxation that lifts every student above the satisfaction floor is a
    single giant group — the "everyone in one of four groups" pathology. Distinct
    Stamgroepen keep the clique family out of it, so only the group-size families
    govern the spread.
    """
    students = []
    for i in range(n_students):
        preferences = (
            []
            if i == 0
            else [Preference(target="Kind0", weight=1.0, kind=PreferenceKind.TOGETHER)]
        )
        students.append(
            StudentEntry(
                f"Kind{i}",
                "Jongen" if i % 2 else "Meisje",
                f"sg{i}",
                None,
                preferences=preferences,
            )
        )
    keys = [g.lower() for g in groups]
    target_groups = GroupCounts(
        counts={key: {"Jongens": 0, "Meisjes": 0} for key in keys},
        display=dict(zip(keys, groups)),
    )
    preference_data = build_preference_data(students, all_to_groups=keys)
    return preference_data, target_groups


def _group_size_spread(solution, groups_to) -> int:
    """Largest minus smallest realized group size in a solved assignment."""
    sizes = Counter(solution.assignment.values())
    for group in groups_to:
        sizes.setdefault(group, 0)
    return max(sizes.values()) - min(sizes.values())


def test_diff_total_cap_shrinks_realized_group_size_spread():
    """A diff_total cap reins in the automatic path's runaway group-size spread.

    Uncapped, all twelve students pile into Kind0's group (spread 12) because that
    is the minimal relaxation reaching the floor. Capping ``max_diff_n_students_total``
    at 4 forbids that: the solve still succeeds (a few students drop below the floor,
    a valid tier-2 outcome), but the realized spread is at most 4.
    """
    preference_data, target_groups = _popular_student_scenario(
        12, ["Rood", "Geel", "Blauw", "Groen"]
    )

    uncapped = engine.solve_within_minimal_relaxation(
        preferences=preference_data.preferences,
        students=preference_data.students_info,
        groups_to=target_groups.counts,
        not_together=[],
    )
    assert _group_size_spread(uncapped, target_groups.counts) > 4

    capped = engine.solve_within_minimal_relaxation(
        preferences=preference_data.preferences,
        students=preference_data.students_info,
        groups_to=target_groups.counts,
        not_together=[],
        maxima=BalanceMaxima(max_diff_n_students_total=4),
    )
    assert _group_size_spread(capped, target_groups.counts) <= 4
