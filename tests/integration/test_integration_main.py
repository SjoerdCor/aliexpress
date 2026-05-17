# pylint: disable=too-many-lines  # Perhaps in the future switch to pytest-regression

"""Integration tests for the main functionality of the AliExpress application.

These tests ensure that the student distribution process works correctly
with both small and full datasets, checking the output against expected results."""
import numpy as np
import pandas as pd
import pytest

from aliexpress import errors, problemsolver
from aliexpress.main import distribute_students_once


def test_distribute_students_once_happy_flow_small():
    """Test the student distribution with a small, quick dataset."""
    result = distribute_students_once(
        path_preferences="tests/integration/voorkeuren_small.xlsx",
        path_groups_to="tests/integration/groepen_small.xlsx",
        path_not_together="tests/integration/niet_samen_small.xlsx",
        on_update=lambda msg: None,
        groupbalance=problemsolver.GroupBalance(max_imbalance_boys_girls_total=7),
    )

    assert isinstance(result, dict)
    assert "download" in result
    # Ensure the download can be read as an Excel file
    pd.read_excel(result["download"])

    assert "dataframes" in result

    dfs = result["dataframes"]
    assert isinstance(dfs, dict)
    expected_keys = {
        "Groepsindeling",
        "Klassenoverzicht",
        "Overgangsmatrix",
        "Leerlingtevredenheid",
        "VervuldeWensen",
    }
    assert expected_keys.issubset(dfs.keys())
    # Assert content of the DataFrames
    groepsindeling = dfs["Groepsindeling"]
    assert isinstance(groepsindeling, pd.DataFrame)
    df_expected_groepsindeling = get_expected_groepsindeling_small()
    pd.testing.assert_frame_equal(groepsindeling, df_expected_groepsindeling)

    klassenoverzicht = dfs["Klassenoverzicht"]
    assert isinstance(klassenoverzicht, pd.DataFrame)
    df_expected_klassenoverzicht = get_expected_klassenoverzicht_small()
    pd.testing.assert_frame_equal(klassenoverzicht, df_expected_klassenoverzicht)

    overgangsmatrix = dfs["Overgangsmatrix"]
    assert isinstance(overgangsmatrix, pd.DataFrame)
    df_expected_overgangsmatrix = get_expected_overgangsmatrix_small()
    pd.testing.assert_frame_equal(overgangsmatrix, df_expected_overgangsmatrix)

    leerlingtevredenheid = dfs["Leerlingtevredenheid"]
    assert isinstance(leerlingtevredenheid, pd.io.formats.style.Styler)
    df_expected_leerlingtevredenheid = get_expected_leerlingtevredenheid_small()
    pd.testing.assert_frame_equal(
        leerlingtevredenheid.data, df_expected_leerlingtevredenheid
    )

    vervuldewensen = dfs["VervuldeWensen"]
    assert isinstance(vervuldewensen, pd.io.formats.style.Styler)
    df_expected_vervuldewensen = get_expected_vervuldewensen_small()
    pd.testing.assert_frame_equal(vervuldewensen.data, df_expected_vervuldewensen)


def test_distribute_students_once_happy_flow_full():
    """Test the student distribution with a full dataset.

    Checks the sorted satisfaction vector, which is a lexmaxmin invariant: any two
    lexmaxmin optima of the same problem produce identical sorted satisfaction vectors.
    This makes the test robust to solver-specific tie-breaking in group assignments.
    """
    result = distribute_students_once(
        path_preferences="tests/integration/voorkeuren.xlsx",
        path_groups_to="tests/integration/groepen.xlsx",
        path_not_together="tests/integration/niet_samen.xlsx",
        on_update=lambda msg: None,
    )

    assert isinstance(result, dict)
    assert "download" in result
    pd.read_excel(result["download"])

    assert "dataframes" in result
    dfs = result["dataframes"]
    assert isinstance(dfs, dict)
    expected_keys = {
        "Groepsindeling",
        "Klassenoverzicht",
        "Overgangsmatrix",
        "Leerlingtevredenheid",
        "VervuldeWensen",
    }
    assert expected_keys.issubset(dfs.keys())

    leerlingtevredenheid = dfs["Leerlingtevredenheid"]
    assert isinstance(leerlingtevredenheid, pd.io.formats.style.Styler)
    sorted_satisfaction = np.sort(leerlingtevredenheid.data["Tevredenheid"].values)
    np.testing.assert_allclose(
        sorted_satisfaction,
        get_expected_sorted_satisfaction_full(),
        atol=1e-5,
    )


def test_distribute_students_once_happy_flow_infeasible():
    """Test the student distribution with infeasible constraints
    ensuring it raises a FeasibilityError."""
    with pytest.raises(errors.FeasibilityError) as exc:
        distribute_students_once(
            path_preferences="tests/integration/voorkeuren.xlsx",
            path_groups_to="tests/integration/groepen.xlsx",
            path_not_together="tests/integration/niet_samen.xlsx",
            on_update=lambda msg: None,
            max_clique=1,
            max_clique_sex=1,
            max_diff_n_students_year=1,
            max_diff_n_students_total=1,
            max_imbalance_boys_girls_year=1,
            max_imbalance_boys_girls_total=1,
        )
    improvements = [
        "Maximale verschil jongens/meisjes totale groep: 3 (+ 2)",
        "Maximale verschil jongens/meisjes nieuwe jaarlaag: 2 (+ 1)",
        "Maximale verschil groepsgrootte nieuwe jaarlaag: 2 (+ 1)",
        "Maximale groep vanuit eerdere groep: 3 (+ 2)",
        "Maximale groep jongens/meisjes vanuit eerdere groep: 2 (+ 1)",
    ]
    for line in improvements:
        assert line in str(exc.value.context["possible_improvement"])


# pylint: disable=missing-function-docstring
def get_expected_groepsindeling_small():
    data = {
        ("Beren", "Jongen"): ["Bram (D)", "", 1, 19],
        ("Beren", "Meisje"): ["Eva (A)", "", 1, np.nan],
        ("Otters", "Jongen"): ["Daan (A)", "", 1, 20],
        ("Otters", "Meisje"): ["Anna (A)", "Claire (B)", 2, np.nan],
    }
    df = pd.DataFrame(data)
    df.index = pd.Index([1, 2, "#", "Groepsgrootte"])
    df.columns.names = ["Groep", "Jongen/meisje"]
    return df


def get_expected_klassenoverzicht_small():
    data = {
        "Jongen": [1, 6, 1, 7],
        "Meisje": [1, 13, 2, 13],
        "VerschilJongensMeisjes": [0, 7, 1, 6],
        "Groepsgrootte": [2, 19, 3, 20],
    }
    df = pd.DataFrame(data)
    df.index = pd.Index(
        [
            ("Beren", "Jaarlaag"),
            ("Beren", "Totaal"),
            ("Otters", "Jaarlaag"),
            ("Otters", "Totaal"),
        ]
    )
    return df


def get_expected_overgangsmatrix_small():
    data = {
        "Beren": [1, 0, 1],
        "Otters": [2, 1, 0],
    }
    df = pd.DataFrame(data)
    df.index = pd.Index(["A", "B", "D"])
    df.index.names = ["Stamgroep"]
    df.columns.names = ["Group"]
    return df


def get_expected_leerlingtevredenheid_small():
    data = {
        "Tevredenheid": [0.9032258064516, 1.0, 0.9448818897637, 0.9376144548895, 0.0],
        "Aantal gehonoreerde wensen": [3.0, 3.0, 4.0, 4.0, 0.0],
        "Aantal wensen": [5.0, 3.0, 7.0, 13.0, 2.0],
    }
    df = pd.DataFrame(data)
    df.index = pd.Index(["Anna", "Bram", "Claire", "Daan", "Eva"])
    df.index.names = ["Leerling"]
    return df


def get_expected_vervuldewensen_small():
    data = {
        ("MinimaleTevredenheid", np.nan, np.nan): [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        ("Jongen/meisje", np.nan, np.nan): [
            "Meisje",
            "Jongen",
            "Meisje",
            "Jongen",
            "Meisje",
        ],
        ("Stamgroep", np.nan, np.nan): ["A", "D", "B", "A", "A"],
        ("Graag met", 1.0, "Waarde"): ["Otters", "Eva", "Eva", "Eva", "Anna"],
        ("Graag met", 1.0, "Gewicht"): [1.0, 3.0, 2.0, 3.0, 2.0],
        ("Graag met", 2.0, "Waarde"): ["Beren", np.nan, "Daan", "Beren", np.nan],
        ("Graag met", 2.0, "Gewicht"): [1.0, np.nan, 1.0, 2.0, np.nan],
        ("Graag met", 3.0, "Waarde"): ["Bram", np.nan, "Beren", "Anna", np.nan],
        ("Graag met", 3.0, "Gewicht"): [1.0, np.nan, 1.0, 3.0, np.nan],
        ("Graag met", 4.0, "Waarde"): ["Daan", np.nan, "Anna", "Bram", np.nan],
        ("Graag met", 4.0, "Gewicht"): [1.0, np.nan, 2.0, 3.0, np.nan],
        ("Graag met", 5.0, "Waarde"): ["Claire", np.nan, "Otters", "Otters", np.nan],
        ("Graag met", 5.0, "Gewicht"): [1.0, np.nan, 1.0, 2.0, np.nan],
        ("Liever niet met", 1.0, "Waarde"): ["Eva", np.nan, np.nan, "Claire", np.nan],
        ("Liever niet met", 1.0, "Gewicht"): [3.0, np.nan, np.nan, 1.0, np.nan],
        ("Niet in", 1.0, "Waarde"): [np.nan, np.nan, np.nan, np.nan, "Otters"],
        ("Niet in", 2.0, "Waarde"): [np.nan, np.nan, np.nan, np.nan, np.nan],
    }
    df = pd.DataFrame(data)
    df.index = pd.Index(["Anna", "Bram", "Claire", "Daan", "Eva"])
    df.index.names = ["Leerling"]
    df.columns.names = ["TypeWens", "Nr", "TypeWaarde"]
    return df


def get_expected_sorted_satisfaction_full():
    return [
        0.5484791673189,
        0.5714285714286,
        0.5714285714286,
        0.5714285714286,
        0.5714285714286,
        0.6567076666989,
        0.6666666666667,
        0.6666666666667,
        0.6666666666667,
        0.6666666666667,
        0.6666666666667,
        0.6763367534524,
        0.709124996087,
        0.7741935483871,
        0.7741935483871,
        0.7846782049872,
        0.7852627661454,
        0.8,
        0.8,
        0.8,
        0.8418251890711,
        0.8571428571429,
        0.8571428571429,
        0.8571428571429,
        0.861287180051,
        0.861287180051,
        0.8947718513662,
        0.9032258064516,
        0.9322107953162,
        0.9333333333333,
        0.9448818897637,
        0.9763779527559,
        0.9763779527559,
        0.984126984127,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
