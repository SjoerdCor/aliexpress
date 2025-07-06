import pytest
import pandas as pd
import numpy as np
from pandas import MultiIndex
from unittest.mock import patch, MagicMock

from aliexpress import errors, datareader


def test_validate_columns_success():
    df = pd.DataFrame(columns=["A", "B", "C"])
    datareader.validate_columns(df, ["A", "B", "C"], "test")


def test_validate_columns_failure():
    df = pd.DataFrame(columns=["A", "B"])
    with pytest.raises(errors.ValidationError) as exc:
        datareader.validate_columns(df, ["A", "B", "C"], "test")
    expected = "Wrong columns for test: \nmissing={'C'}\nextra=set()"
    assert str(exc.value) == expected


def test_check_mandatory_columns_success():
    df = pd.DataFrame({"A": [1], "B": [2]})
    datareader.check_mandatory_columns(df, ["A", "B"], "test")


def test_check_mandatory_columns_missing_data():
    df = pd.DataFrame({"A": [np.nan], "B": [2]})
    df.index = pd.Index([np.nan], name="idx")
    with pytest.raises(errors.ValidationError) as exc:
        datareader.check_mandatory_columns(df, ["A", "B"], "test")
    assert "empty_mandatory_columns" in exc.value.code


def test_toggle_negative_weights():
    df = pd.DataFrame(
        {
            "Leerling": ["John", "Jane"],
            "TypeWens": ["Liever niet met", "Graag met"],
            "Gewicht": [-1, 2],
        }
    )
    df.set_index(["Leerling", "TypeWens"], inplace=True)
    result = datareader.toggle_negative_weights(df)
    assert all(result["Gewicht"] > 0)
    assert set(result.index.get_level_values("TypeWens")) == {
        "Graag met",
    }


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("  John  ", "John"),
        ("<script>", "Script"),
        ("ANNa-MAriE", "Anna-Marie"),
        ("Anne marie", "AnneMarie"),
        (42, 42),  # not a string
    ],
)
def test_clean_name(input_str, expected):
    assert datareader.clean_name(input_str) == expected


@patch("aliexpress.datareader.pd.read_excel")
def test_voorkeuren_processor_init(mock_read_excel):
    mock_df = pd.DataFrame(
        [
            ["Graag met", "Graag met", "Graag met", "Liever niet met"],
            [1.0, 2.0, 3.0, 1.0],
            ["Waarde", "Gewicht", "Waarde", "Waarde"],
            ["Alice", 1, "Bob", "Charlie"],
        ],
        columns=[0, 1, 2, 3],
        index=["Leerling", "Leerling", "Leerling", "John"],
    )
    mock_read_excel.return_value = mock_df
    with patch("aliexpress.datareader.VoorkeurenProcessor._validate_input"):
        processor = datareader.VoorkeurenProcessor("dummy.xlsx")
        assert isinstance(processor.df, pd.DataFrame)


def test_voorkeuren_processor_clean_input():
    df = pd.DataFrame(
        {("A", "B", "C"): ["  john ", "<script>"]}, index=[" alice ", "bob"]
    )
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)  # bypass __init__
    cleaned = processor.clean_input(df)
    assert "John" in cleaned.iloc[:, 0].values
    assert "Script" in cleaned.iloc[:, 0].values
    assert "Alice" in cleaned.index


@patch("aliexpress.datareader.pd.read_excel")
def test_read_not_together_success(mock_read_excel):
    data = {
        "Max aantal samen": [2],
        "Leerling 1": ["Alice"],
        "Leerling 2": ["Bob"],
    }
    for llnr in range(3, 13):
        data[f'Leerling {llnr}'] = pd.NA

    mock_read_excel.return_value = pd.DataFrame(data)
    students = ["Alice", "Bob"]
    result = datareader.read_not_together("dummy.xlsx", students, n_groups=2)
    assert result[0]["Max_aantal_samen"] == 2
    assert result[0]["group"] == {"Alice", "Bob"}


@patch("aliexpress.datareader.pd.read_excel")
def test_read_not_together_duplicate_student_error(mock_read_excel):
    data = {
        "Max aantal samen": [2],
        "Leerling 1": ["Alice"],
        "Leerling 2": ["Alice"],
    }
    for llnr in range(3, 13):
        data[f'Leerling {llnr}'] = pd.NA
    mock_read_excel.return_value = pd.DataFrame(data)
    with pytest.raises(datareader.ValidationError) as exc:
        datareader.read_not_together("dummy.xlsx", ["Alice"], 2)
    assert "duplicated_students_not_together" in exc.value.code


@patch("aliexpress.datareader.pd.read_excel")
def test_read_groups_excel_success(mock_read_excel):
    df = pd.DataFrame(
        {
            "Groepen": ["A"],
            "Jongens": [5],
            "Meisjes": [6],
        }
    )
    mock_read_excel.return_value = df
    result = datareader.read_groups_excel("groups.xlsx")
    assert "A" in result
    assert result["A"]["Jongens"] == 5
    assert result["A"]["Meisjes"] == 6


@patch("aliexpress.datareader.pd.read_excel")
def test_read_groups_excel_empty(mock_read_excel):
    mock_read_excel.return_value = pd.DataFrame(
        columns=["Groepen", "Jongens", "Meisjes"]
    )
    with pytest.raises(errors.ValidationError) as exc:
        datareader.read_groups_excel("groups.xlsx")
    assert "empty_df" in exc.value.code
