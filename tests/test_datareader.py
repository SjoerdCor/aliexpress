import pytest
import pandas as pd
import numpy as np
from pandas import MultiIndex
from unittest.mock import patch, MagicMock

from aliexpress import errors, datareader

@pytest.fixture
def valid_voorkeuren_df():
    header = [
        ("MinimaleTevredenheid", np.nan, np.nan),
        ("Jongen/meisje", np.nan, np.nan),
        ("Stamgroep", np.nan, np.nan),
        ("Graag met", 1, "Waarde"), ("Graag met", 1, "Gewicht"),
        ("Graag met", 2, "Waarde"), ("Graag met", 2, "Gewicht"),
        ("Graag met", 3, "Waarde"), ("Graag met", 3, "Gewicht"),
        ("Graag met", 4, "Waarde"), ("Graag met", 4, "Gewicht"),
        ("Graag met", 5, "Waarde"), ("Graag met", 5, "Gewicht"),
        ("Liever niet met", 1, "Waarde"), ("Liever niet met", 1, "Gewicht"),
        ("Niet in", 1, "Waarde"), ("Niet in", 2, "Waarde"),
    ]
    columns = pd.MultiIndex.from_tuples(header, names=["TypeWens", "Nr", "TypeWaarde"])
    data = [[0.5, "Jongen", "A" , "Jane", 1, "Alice", 2, "Blauw", 0.5, np.nan, np.nan, np.nan, np.nan, "Eve", 2, "Oranje", np.nan],
            [np.nan, "Meisje", "B", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, "Meisje", "B", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, "Meisje", "B", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],

            ]
    
    df = pd.DataFrame(data, columns=columns, index=pd.Index(["John", "Jane", "Alice", "Eve"], name="Leerling"))
    return df

@pytest.fixture
def valid_groepen_df():
    return pd.DataFrame({"Groepen": ["A"], "Jongens": [5], "Meisjes": [6]})

@pytest.fixture
def valid_niet_samen_df():
    data = {"Max aantal samen": [2], "Leerling 1": ["Alice"], "Leerling 2": ["Bob"]}
    for llnr in range(3, 13):
        data[f'Leerling {llnr}'] = pd.NA
    return pd.DataFrame(data)


def test_validate_columns_success():
    df = pd.DataFrame(columns=["A", "B", "C"])
    datareader.validate_columns(df, ["A", "B", "C"], "test")


def test_validate_columns_extra_and_missing():
    df = pd.DataFrame(columns=["A", "B", "D"])
    with pytest.raises(errors.ValidationError) as exc:
        datareader.validate_columns(df, ["A", "B", "C"], "test")
    expected = "Wrong columns for test: \nmissing={'C'}\nextra={'D'}"
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

def test_check_mandatory_columns_index_nan():
    df = pd.DataFrame({"A": [1], "B": [2]})
    df.index = pd.Index([float('nan')], name="idx")
    with pytest.raises(errors.ValidationError) as exc:
        datareader.check_mandatory_columns(df, ["A", "B"], "test")
    assert "empty_mandatory_columns_test" in exc.value.code


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

def test_toggle_negative_weights_liever_niet_met():
    df = pd.DataFrame({
        "Leerling": ["John", "Jane"],
        "TypeWens": ["Liever niet met", "Graag met"],
        "Gewicht": [1, 2],
    })
    df.set_index(["Leerling", "TypeWens"], inplace=True)
    result = datareader.toggle_negative_weights(df, mask="Liever niet met")
    assert result["Gewicht"].tolist() == [-1, 2]

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


def test_voorkeuren_processor_validate_input_duplicate(valid_voorkeuren_df):
    df = pd.concat([valid_voorkeuren_df, valid_voorkeuren_df])
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    with pytest.raises(datareader.ValidationError) as exc:
        processor._validate_input(df)
    assert "duplicate_students_preferences" in exc.value.code


def test_voorkeuren_processor_validate_input_wrong_sex(valid_voorkeuren_df):
    df = valid_voorkeuren_df.copy()
    df.iloc[0, df.columns.get_loc(("Jongen/meisje", np.nan, np.nan))] = "Alien"
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    with pytest.raises(datareader.ValidationError) as exc:
        processor._validate_input(df)
    assert "wrong_sex" in exc.value.code


def test_voorkeuren_processor_validate_input_duplicated_values(valid_voorkeuren_df):
    df = valid_voorkeuren_df.copy()
    print(df)
    df.loc["John", ("Graag met", 1, "Waarde")] = "Bob"
    df.loc["John", ("Graag met", 2, "Waarde")] = "Bob"
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    with pytest.raises(datareader.ValidationError) as exc:
        processor._validate_input(df)
    assert "duplicated_values_preferences" in exc.value.code


def test_voorkeuren_processor_restructure_and_validate_preferences(valid_voorkeuren_df):
    df = valid_voorkeuren_df.copy()
    df.loc["John", ("Graag met", 1, "Gewicht")] = -1

    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    processor.input = df
    processor.df = df.copy()
    processor.restructure()
    
    with pytest.raises(datareader.ValidationError) as exc:
        processor.validate_preferences(["Oranje", "Blauw"])
    assert "negative_weights_preferences" in exc.value.code

def test_voorkeuren_processor_validate_preferences_wrong_index():
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    processor.df = pd.DataFrame({"Gewicht": [1], "Waarde": ["A"]})
    with pytest.raises(datareader.ValidationError) as exc:
        processor.validate_preferences()
    assert "wrong_index_names_preferences" in exc.value.code

def test_voorkeuren_processor_validate_preferences_invalid_values(valid_voorkeuren_df):
    df = valid_voorkeuren_df.copy()

    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    processor.input = df
    processor.df = df.copy()
    processor.restructure()

    with pytest.raises(datareader.ValidationError) as exc:
        processor.validate_preferences(["Blauw"])
    assert "invalid_values_preferences" in exc.value.code

def test_voorkeuren_processor_process_and_get_students_meta_info(valid_voorkeuren_df):

    df = valid_voorkeuren_df.copy()
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    processor.input = df
    processor.df = df.copy()

    meta = processor.get_students_meta_info()
    expected = {'John': {'MinimaleTevredenheid': 0.5,
  'Jongen/meisje': 'Jongen',
  'Stamgroep': 'A'},
 'Jane': {'MinimaleTevredenheid': float('nan'),
  'Jongen/meisje': 'Meisje',
  'Stamgroep': 'B'},
 'Alice': {'MinimaleTevredenheid': float('nan'),
  'Jongen/meisje': 'Meisje',
  'Stamgroep': 'B'},
 'Eve': {'MinimaleTevredenheid': float('nan'),
  'Jongen/meisje': 'Meisje',
  'Stamgroep': 'B'}}

    def dicts_equal_with_nan(d1, d2):
        if d1.keys() != d2.keys():
            return False
        for k in d1:
            v1, v2 = d1[k], d2[k]
            if v1.keys() != v2.keys():
                return False
            for subk in v1:
                val1, val2 = v1[subk], v2[subk]
                if isinstance(val1, float) and isinstance(val2, float) and np.isnan(val1) and np.isnan(val2):
                    continue
                if val1 != val2:
                    return False
        return True

    assert dicts_equal_with_nan(meta, expected)

def test_read_not_together_unknown_student():
    df = pd.DataFrame({
        "Max aantal samen": [2],
        "Leerling 1": ["Alice"],
        "Leerling 2": ["Unknown"],
    })
    for llnr in range(3, 13):
        df[f'Leerling {llnr}'] = pd.NA
    with patch("aliexpress.datareader.pd.read_excel", return_value=df):
        with pytest.raises(datareader.ValidationError) as exc:
            datareader.read_not_together("dummy.xlsx", ["Alice"], 2)
        assert "unknown_students_not_together" in exc.value.code

def test_read_not_together_too_strict():
    df = pd.DataFrame({
        "Max aantal samen": [1],
        "Leerling 1": ["Alice"],
        "Leerling 2": ["Bob"],
        "Leerling 3": ["Charlie"],
    })
    for llnr in range(4, 13):
        df[f'Leerling {llnr}'] = pd.NA
    with patch("aliexpress.datareader.pd.read_excel", return_value=df):
        with pytest.raises(datareader.ValidationError) as exc:
            datareader.read_not_together("dummy.xlsx", ["Alice", "Bob", "Charlie"], 2)
        assert "too_strict_not_together" in exc.value.code
