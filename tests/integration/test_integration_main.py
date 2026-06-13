"""Integration tests for the main functionality of the AliExpress application.

These tests ensure the student distribution process works on both a small and a full
dataset. The exact group *labels* of an assignment are a degenerate optimum (the groups
can be relabelled, e.g. "Blauw" <-> "Geel", without changing who sits with whom), so the
group-placement tables are checked on their structure and respected class balance rather
than cell-by-cell. The per-student satisfaction - the actual optimization objective - is
uniquely determined and is asserted in full."""

import re

import pandas as pd
import pytest

from aliexpress import errors
from aliexpress.main import distribute_students_once
from aliexpress.problemsolver import GroupBalance

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
    "Groepsindeling",
    "Klassenoverzicht",
    "Overgangsmatrix",
    "Leerlingtevredenheid",
    "VervuldeWensen",
}

_SMALL_SATISFACTION = {
    "Tevredenheid": {
        "Anna": 0.903226,
        "Bram": 1.0,
        "Claire": 0.944882,
        "Daan": 0.937614,
        "Eva": 0.0,
    },
    "Aantal gehonoreerde wensen": {
        "Anna": 3.0,
        "Bram": 3.0,
        "Claire": 4.0,
        "Daan": 4.0,
        "Eva": 0.0,
    },
    "Aantal wensen": {
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
        "Anne": 0.785263,
        "Anne Claire": 0.571429,
        "Benjamin": 0.738796,
        "Bram": 0.774194,
        "Cas": 0.857143,
        "Daan": 1.0,
        "David": 0.857143,
        "Eline": 0.932211,
        "Emily": 0.976378,
        "Esmee": 0.607369,
        "Feline": 0.738796,
        "Fenna": 0.861287,
        "Iris": 0.571429,
        "Jack": 0.861287,
        "Jayden": 0.784678,
        "Jill": 1.0,
        "Julia": 0.984127,
        "Julian": 0.709125,
        "Jurre": 0.666667,
        "Lars": 0.666667,
        "Liv A.": 0.661054,
        "Liv B.": 0.656708,
        "Lois": 1.0,
        "Lotte": 0.784678,
        "Lucas": 0.571429,
        "Lynn": 1.0,
        "Mats": 0.88189,
        "Max": 0.894772,
        "Naomi": 1.0,
        "Nina": 0.533333,
        "Noor": 0.571429,
        "Nora": 0.933333,
        "Siem X.": 0.666667,
        "Siem Y.": 0.709125,
        "Sophie": 1.0,
        "Stijn": 0.8,
        "Sven": 0.666667,
        "Tijn": 0.666667,
        "Vera": 0.533333,
        "Zoe": 0.979573,
    },
    "Aantal gehonoreerde wensen": {
        "Adam": 1.0,
        "Amy": 1.5,
        "Anna": 5.0,
        "Anne": 1.5,
        "Anne Claire": 1.0,
        "Benjamin": 1.5,
        "Bram": 2.0,
        "Cas": 2.0,
        "Daan": 2.0,
        "David": 2.0,
        "Eline": 3.5,
        "Emily": 5.0,
        "Esmee": 1.0,
        "Feline": 1.5,
        "Fenna": 2.5,
        "Iris": 1.0,
        "Jack": 2.5,
        "Jayden": 2.0,
        "Jill": 2.0,
        "Julia": 5.0,
        "Julian": 1.5,
        "Jurre": 1.0,
        "Lars": 1.0,
        "Liv A.": 1.5,
        "Liv B.": 1.5,
        "Lois": 1.0,
        "Lotte": 2.0,
        "Lucas": 1.0,
        "Lynn": 1.0,
        "Mats": 3.0,
        "Max": 3.0,
        "Naomi": 1.0,
        "Nina": 1.0,
        "Noor": 1.0,
        "Nora": 3.0,
        "Siem X.": 1.0,
        "Siem Y.": 1.5,
        "Sophie": 2.0,
        "Stijn": 2.0,
        "Sven": 1.0,
        "Tijn": 1.0,
        "Vera": 1.0,
        "Zoe": 5.0,
    },
    "Aantal wensen": {
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
    """Assert the result shape and return the five output tables."""
    assert isinstance(result, dict)
    assert "download" in result
    pd.read_excel(result["download"])  # download must be a readable Excel file

    assert "dataframes" in result
    dfs = result["dataframes"]
    assert isinstance(dfs, dict)
    assert _EXPECTED_KEYS.issubset(dfs.keys())

    assert isinstance(dfs["Groepsindeling"], pd.DataFrame)
    assert isinstance(dfs["Klassenoverzicht"], pd.DataFrame)
    assert isinstance(dfs["Overgangsmatrix"], pd.DataFrame)
    assert isinstance(dfs["Leerlingtevredenheid"], pd.io.formats.style.Styler)
    assert isinstance(dfs["VervuldeWensen"], pd.io.formats.style.Styler)
    return dfs


def _assert_consistency(dfs, groups, stamgroepen):
    """Structural + counting invariants that hold for any optimal assignment."""
    groep = dfs["Groepsindeling"]
    assert groep.columns.names == ["Groep", "Jongen/meisje"]
    assert set(groep.columns.get_level_values("Groep")) == groups

    klas = dfs["Klassenoverzicht"]
    assert list(klas.columns) == [
        "Jongen",
        "Meisje",
        "VerschilJongensMeisjes",
        "Groepsgrootte",
    ]
    assert (klas["Jongen"] + klas["Meisje"] == klas["Groepsgrootte"]).all()
    assert (
        (klas["Jongen"] - klas["Meisje"]).abs() == klas["VerschilJongensMeisjes"]
    ).all()

    trans = dfs["Overgangsmatrix"]
    assert trans.index.name == "Stamgroep"
    assert set(trans.columns) == groups
    assert set(trans.index) == stamgroepen

    tevr = dfs["Leerlingtevredenheid"].data
    wensen = dfs["VervuldeWensen"].data
    # Every student appears once in the satisfaction tables and is placed exactly once.
    n_students = len(tevr)
    assert set(wensen.index) == set(tevr.index)
    assert trans.to_numpy().sum() == n_students
    jaar = klas[klas.index.get_level_values(1) == "Jaarlaag"]
    assert jaar["Groepsgrootte"].sum() == n_students
    return tevr, klas, trans


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
        on_update=lambda msg: None,
        groupbalance=GroupBalance(max_imbalance_boys_girls_total=7),
    )
    dfs = _tables(result)
    tevr, klas, trans = _assert_consistency(dfs, {"Beren", "Otters"}, {"A", "B", "D"})
    _assert_satisfaction(tevr, _SMALL_SATISFACTION)

    totaal = klas[klas.index.get_level_values(1) == "Totaal"]
    jaar = klas[klas.index.get_level_values(1) == "Jaarlaag"]
    # Manual balance: GroupBalance(max_imbalance_boys_girls_total=7) + loose defaults.
    assert trans.to_numpy().max() <= 5  # max_clique
    assert totaal["VerschilJongensMeisjes"].max() <= 7  # max_imbalance_boys_girls_total
    assert jaar["VerschilJongensMeisjes"].max() <= 2  # max_imbalance_boys_girls_year
    assert totaal["Groepsgrootte"].max() - totaal["Groepsgrootte"].min() <= 3
    assert jaar["Groepsgrootte"].max() - jaar["Groepsgrootte"].min() <= 2


def test_distribute_students_once_happy_flow_full():
    """Full dataset, satisfaction maximized within the auto-determined minimal balance
    relaxation that still lets every student fulfil at least one positive wish."""
    result = distribute_students_once(
        path_preferences="tests/integration/voorkeuren.xlsx",
        path_groups_to="tests/integration/groepen.xlsx",
        not_together=_NOT_TOGETHER_FULL,
        on_update=lambda msg: None,
    )
    dfs = _tables(result)
    tevr, klas, trans = _assert_consistency(
        dfs,
        {"Blauw", "Geel", "Groen", "Oranje"},
        {"Kaboutertuin", "Torteltuin", "Tovertuin", "Vlindertuin"},
    )
    _assert_satisfaction(tevr, _FULL_SATISFACTION)
    assert (tevr["Tevredenheid"] > 0).all()  # the goal: every student ends up positive

    totaal = klas[klas.index.get_level_values(1) == "Totaal"]
    jaar = klas[klas.index.get_level_values(1) == "Jaarlaag"]
    # Class balance realized within the auto-determined minimal relaxation.
    assert trans.to_numpy().max() <= 3  # max students from one stamgroep in a group
    assert totaal["VerschilJongensMeisjes"].max() <= 2
    assert jaar["VerschilJongensMeisjes"].max() <= 3
    assert totaal["Groepsgrootte"].max() - totaal["Groepsgrootte"].min() <= 2
    assert jaar["Groepsgrootte"].max() - jaar["Groepsgrootte"].min() <= 1


def test_distribute_students_once_happy_flow_infeasible():
    """Infeasible constraints must raise a FeasibilityError with a relaxation suggestion."""
    with pytest.raises(errors.FeasibilityError) as exc:
        distribute_students_once(
            path_preferences="tests/integration/voorkeuren.xlsx",
            path_groups_to="tests/integration/groepen.xlsx",
            not_together=_NOT_TOGETHER_FULL,
            on_update=lambda msg: None,
            groupbalance=GroupBalance(
                max_clique=1,
                max_clique_sex=1,
                max_diff_n_students_year=1,
                max_diff_n_students_total=1,
                max_imbalance_boys_girls_year=1,
                max_imbalance_boys_girls_total=1,
            ),
        )
    # Parse "<label>: <new value> (+ <relaxation>)" lines into {label: relaxation}.
    msg = str(exc.value.context["possible_improvement"])
    relaxations = {
        m.group("label"): int(m.group("relax"))
        for m in re.finditer(r"(?P<label>.+?): \d+ \(\+ (?P<relax>\d+)\)", msg)
    }
    # The problem needs a fixed minimal total relaxation (7 units) to become feasible.
    # How that budget is split across the individual limits is a degenerate, non-unique
    # optimum (the solver may shift it between limits), so only the total - the
    # solver-independent invariant - is asserted, plus that a suggestion is produced.
    assert relaxations
    assert sum(relaxations.values()) == 7
