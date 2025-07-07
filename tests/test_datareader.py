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
        ("Graag met", 1.0, "Waarde"),
        ("Graag met", 1.0, "Gewicht"),
        ("Graag met", 2.0, "Waarde"),
        ("Graag met", 2.0, "Gewicht"),
        ("Graag met", 3.0, "Waarde"),
        ("Graag met", 3.0, "Gewicht"),
        ("Graag met", 4.0, "Waarde"),
        ("Graag met", 4.0, "Gewicht"),
        ("Graag met", 5.0, "Waarde"),
        ("Graag met", 5.0, "Gewicht"),
        ("Liever niet met", 1.0, "Waarde"),
        ("Liever niet met", 1.0, "Gewicht"),
        ("Niet in", 1.0, "Waarde"),
        ("Niet in", 2.0, "Waarde"),
    ]
    columns = pd.MultiIndex.from_tuples(header, names=["TypeWens", "Nr", "TypeWaarde"])
    data = [
        [
            0.5,
            "Jongen",
            "A",
            "Jane",
            1,
            "Alice",
            2,
            "Blauw",
            0.5,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            "Eve",
            2,
            "Oranje",
            np.nan,
        ],
        [
            np.nan,
            "Meisje",
            "B",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            np.nan,
            "Meisje",
            "B",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            np.nan,
            "Meisje",
            "B",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
    ]

    df = pd.DataFrame(
        data,
        columns=columns,
        index=pd.Index(["John", "Jane", "Alice", "Eve"], name="Leerling"),
    )
    return df


@pytest.fixture
def valid_groepen_df():
    return pd.DataFrame({"Groepen": ["A"], "Jongens": [5], "Meisjes": [6]})


@pytest.fixture
def valid_niet_samen_df():
    data = {"Max aantal samen": [2], "Leerling 1": ["Alice"], "Leerling 2": ["Bob"]}
    for llnr in range(3, 13):
        data[f"Leerling {llnr}"] = pd.NA
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
    df.index = pd.Index([1], name="idx")
    with pytest.raises(errors.ValidationError) as exc:
        datareader.check_mandatory_columns(df, ["A", "B"], "test")
    assert "empty_mandatory_columns" in exc.value.code


def test_check_mandatory_columns_index_nan():
    df = pd.DataFrame({"A": [1], "B": [2]})
    df.index = pd.Index([float("nan")], name="idx")
    with pytest.raises(errors.ValidationError) as exc:
        datareader.check_mandatory_columns(df, ["A", "B"], "test")
    assert "empty_mandatory_columns_test" in exc.value.code


def test_toggle_negative_weights():
    df = pd.DataFrame(
        {
            "Leerling": ["John", "Jane"],
            "TypeWens": ["Graag met", "Graag met"],
            "Gewicht": [-1, 2],
        }
    )
    df.set_index(["Leerling", "TypeWens"], inplace=True)
    result = datareader.toggle_negative_weights(df)
    assert result["Gewicht"].tolist() == [1, 2]
    expected = ["Liever niet met", "Graag met"]
    assert result.index.get_level_values("TypeWens").tolist() == expected


def test_toggle_negative_weights_liever_niet_met():
    df = pd.DataFrame(
        {
            "Leerling": ["John", "Jane"],
            "TypeWens": ["Liever niet met", "Graag met"],
            "Gewicht": [1, 2],
        }
    )
    df.set_index(["Leerling", "TypeWens"], inplace=True)
    result = datareader.toggle_negative_weights(df, mask="Liever niet met")
    assert result["Gewicht"].tolist() == [-1, 2]
    expected = ["Graag met", "Graag met"]
    assert result.index.get_level_values("TypeWens").tolist() == expected


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
def test_voorkeuren_processor_init(mock_read_excel, valid_voorkeuren_df):
    index = pd.Index(
        ["Leerling", np.nan, np.nan, "John", "Jane", "Alice", "Eve"],
        dtype="object",
        name="Leerling",
    )

    mock_df = pd.DataFrame(
        [
            {
                1: "MinimaleTevredenheid",
                2: "Jongen/meisje",
                3: "Stamgroep",
                4: "Graag met",
                5: np.nan,
                6: "Graag met",
                7: np.nan,
                8: "Graag met",
                9: np.nan,
                10: "Graag met",
                11: np.nan,
                12: "Graag met",
                13: np.nan,
                14: "Liever niet met",
                15: np.nan,
                16: "Niet in",
                17: "Niet in",
            },
            {
                1: np.nan,
                2: np.nan,
                3: np.nan,
                4: 1,
                5: np.nan,
                6: 2,
                7: np.nan,
                8: 3,
                9: np.nan,
                10: 4,
                11: np.nan,
                12: 5,
                13: np.nan,
                14: 1,
                15: np.nan,
                16: 1,
                17: 2,
            },
            {
                1: np.nan,
                2: np.nan,
                3: np.nan,
                4: "Naam (leerling of stamgroep)",
                5: "Gewicht",
                6: "Naam (leerling of stamgroep)",
                7: "Gewicht",
                8: "Naam (leerling of stamgroep)",
                9: "Gewicht",
                10: "Naam (leerling of stamgroep)",
                11: "Gewicht",
                12: "Naam (leerling of stamgroep)",
                13: "Gewicht",
                14: "Naam (leerling of stamgroep)",
                15: "Gewicht",
                16: "Stamgroep",
                17: "Stamgroep",
            },
            {
                1: 0.5,
                2: "Jongen",
                3: "A",
                4: "Jane",
                5: 1,
                6: "Alice",
                7: 2,
                8: "Blauw",
                9: 0.5,
                10: np.nan,
                11: np.nan,
                12: np.nan,
                13: np.nan,
                14: "Eve",
                15: 2,
                16: "Oranje",
                17: np.nan,
            },
            {
                1: np.nan,
                2: "Meisje",
                3: "B",
                4: np.nan,
                5: np.nan,
                6: np.nan,
                7: np.nan,
                8: np.nan,
                9: np.nan,
                10: np.nan,
                11: np.nan,
                12: np.nan,
                13: np.nan,
                14: np.nan,
                15: np.nan,
                16: np.nan,
                17: np.nan,
            },
            {
                1: np.nan,
                2: "Meisje",
                3: "B",
                4: np.nan,
                5: np.nan,
                6: np.nan,
                7: np.nan,
                8: np.nan,
                9: np.nan,
                10: np.nan,
                11: np.nan,
                12: np.nan,
                13: np.nan,
                14: np.nan,
                15: np.nan,
                16: np.nan,
                17: np.nan,
            },
            {
                1: np.nan,
                2: "Meisje",
                3: "B",
                4: np.nan,
                5: np.nan,
                6: np.nan,
                7: np.nan,
                8: np.nan,
                9: np.nan,
                10: np.nan,
                11: np.nan,
                12: np.nan,
                13: np.nan,
                14: np.nan,
                15: np.nan,
                16: np.nan,
                17: np.nan,
            },
        ],
        index=index,
    )
    mock_read_excel.return_value = mock_df
    expected = valid_voorkeuren_df.copy()
    with patch("aliexpress.datareader.VoorkeurenProcessor._validate_input"):
        processor = datareader.VoorkeurenProcessor("dummy.xlsx")
        assert isinstance(processor.df, pd.DataFrame)
        assert processor.df.equals(processor.input)
        pd.testing.assert_frame_equal(processor.df, expected)


def test_voorkeuren_processor_wrong_columns(valid_voorkeuren_df):
    df = valid_voorkeuren_df.copy()
    df = df.iloc[:, :-1]
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    with pytest.raises(errors.ValidationError) as exc:
        processor._validate_input(df.iloc[:, :-1])
    assert "wrong_columns_preferences" in exc.value.code


def test_voorkeuren_processor_empty_df(valid_voorkeuren_df):
    df = valid_voorkeuren_df.copy()
    df = df.iloc[:0, :]
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    with pytest.raises(errors.ValidationError) as exc:
        processor._validate_input(df)
    assert "empty_df" in exc.value.code


def test_voorkeuren_processor_mandatory_columns(valid_voorkeuren_df):
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)

    df = valid_voorkeuren_df.copy()
    df["Stamgroep"] = np.nan
    with pytest.raises(errors.ValidationError) as exc:
        processor._validate_input(df)
    assert "empty_mandatory_columns_preferences" in exc.value.code

    df = valid_voorkeuren_df.copy()
    df["Jongen/meisje"] = np.nan
    with pytest.raises(errors.ValidationError) as exc:
        processor._validate_input(df)
    assert "empty_mandatory_columns_preferences" in exc.value.code


def test_voorkeuren_processor_clean_input():
    df = pd.DataFrame(
        {("A", "B", "C"): ["  john ", "<script>"]}, index=[" alice ", "bob"]
    )
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    cleaned = processor.clean_input(df)
    assert "John" in cleaned.iloc[:, 0].values
    assert "Script" in cleaned.iloc[:, 0].values
    assert "Alice" in cleaned.index


def test_voorkeuren_processor_process(valid_voorkeuren_df):
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    processor.input = valid_voorkeuren_df
    processor.df = valid_voorkeuren_df.copy()
    processor.restructure()

    expected = pd.DataFrame(
        {
            "Waarde": {
                ("John", "Graag met", 1.0): "Jane",
                ("John", "Graag met", 2.0): "Alice",
                ("John", "Graag met", 3.0): "Blauw",
                ("John", "Liever niet met", 1.0): "Eve",
                ("John", "Niet in", 1.0): "Oranje",
            },
            "Gewicht": {
                ("John", "Graag met", 1.0): 1.0,
                ("John", "Graag met", 2.0): 2.0,
                ("John", "Graag met", 3.0): 0.5,
                ("John", "Liever niet met", 1.0): 2.0,
                ("John", "Niet in", 1.0): 1.0,
            },
        }
    )
    expected.index.names = ["Leerling", "TypeWens", "Nr"]
    expected.columns.names = ["TypeWaarde"]
    pd.testing.assert_frame_equal(processor.df, expected)


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
    df.loc["John", ("Graag met", 1, "Waarde")] = "Jane"
    df.loc["John", ("Graag met", 2, "Waarde")] = "Jane"
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
    expected = {
        "John": {
            "MinimaleTevredenheid": 0.5,
            "Jongen/meisje": "Jongen",
            "Stamgroep": "A",
        },
        "Jane": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Meisje",
            "Stamgroep": "B",
        },
        "Alice": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Meisje",
            "Stamgroep": "B",
        },
        "Eve": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Meisje",
            "Stamgroep": "B",
        },
    }

    def dicts_equal_with_nan(d1, d2):
        if d1.keys() != d2.keys():
            return False
        for k in d1:
            v1, v2 = d1[k], d2[k]
            if v1.keys() != v2.keys():
                return False
            for subk in v1:
                val1, val2 = v1[subk], v2[subk]
                if (
                    isinstance(val1, float)
                    and isinstance(val2, float)
                    and np.isnan(val1)
                    and np.isnan(val2)
                ):
                    continue
                if val1 != val2:
                    return False
        return True

    assert dicts_equal_with_nan(meta, expected)


@patch("aliexpress.datareader.pd.read_excel")
def test_read_not_together_success(mock_read_excel):
    data = {
        "Max aantal samen": [2],
        "Leerling 1": ["Alice"],
        "Leerling 2": ["Bob"],
    }
    for llnr in range(3, 13):
        data[f"Leerling {llnr}"] = pd.NA

    mock_read_excel.return_value = pd.DataFrame(data)
    students = ["Alice", "Bob"]
    result = datareader.read_not_together("dummy.xlsx", students, n_groups=2)
    assert result == [{"Max_aantal_samen": 2, "group": {"Alice", "Bob"}}]


def test_read_not_together_incompletely_filled():
    df = pd.DataFrame(
        {
            "Max aantal samen": [2],
            "Leerling 1": ["Alice"],
        }
    )
    for llnr in range(2, 13):
        df[f"Leerling {llnr}"] = pd.NA
    with patch("aliexpress.datareader.pd.read_excel", return_value=df):
        with pytest.raises(datareader.ValidationError) as exc:
            datareader.read_not_together("dummy.xlsx", ["Alice"], 2)
        assert "empty_mandatory_columns_not_together" == exc.value.code


@patch("aliexpress.datareader.pd.read_excel")
def test_read_not_together_duplicate_student_error(mock_read_excel):
    data = {
        "Max aantal samen": [2],
        "Leerling 1": ["Alice"],
        "Leerling 2": ["Alice"],
    }
    for llnr in range(3, 13):
        data[f"Leerling {llnr}"] = pd.NA
    mock_read_excel.return_value = pd.DataFrame(data)
    with pytest.raises(datareader.ValidationError) as exc:
        datareader.read_not_together("dummy.xlsx", ["Alice"], 2)
    assert "duplicated_students_not_together" in exc.value.code


def test_read_not_together_unknown_student():
    df = pd.DataFrame(
        {
            "Max aantal samen": [2],
            "Leerling 1": ["Alice"],
            "Leerling 2": ["Unknown"],
        }
    )
    for llnr in range(3, 13):
        df[f"Leerling {llnr}"] = pd.NA
    with patch("aliexpress.datareader.pd.read_excel", return_value=df):
        with pytest.raises(datareader.ValidationError) as exc:
            datareader.read_not_together("dummy.xlsx", ["Alice"], 2)
        assert "unknown_students_not_together" in exc.value.code


def test_read_not_together_too_strict():
    df = pd.DataFrame(
        {
            "Max aantal samen": [1],
            "Leerling 1": ["Alice"],
            "Leerling 2": ["Bob"],
            "Leerling 3": ["Charlie"],
        }
    )
    for llnr in range(4, 13):
        df[f"Leerling {llnr}"] = pd.NA
    with patch("aliexpress.datareader.pd.read_excel", return_value=df):
        with pytest.raises(datareader.ValidationError) as exc:
            datareader.read_not_together("dummy.xlsx", ["Alice", "Bob", "Charlie"], 2)
        assert "too_strict_not_together" in exc.value.code


@patch("aliexpress.datareader.pd.read_excel")
def test_read_groups_excel_success(mock_read_excel):
    df = pd.DataFrame(
        {
            "Groepen": ["De Flamingo's"],
            "Jongens": [5],
            "Meisjes": [6],
        }
    )
    mock_read_excel.return_value = df
    result = datareader.read_groups_excel("groups.xlsx")
    assert result == {"DeFlamingos": {"Jongens": 5, "Meisjes": 6}}


@patch("aliexpress.datareader.pd.read_excel")
def test_read_groups_excel_empty(mock_read_excel):
    mock_read_excel.return_value = pd.DataFrame(
        columns=["Groepen", "Jongens", "Meisjes"]
    )
    with pytest.raises(errors.ValidationError) as exc:
        datareader.read_groups_excel("groups.xlsx")
    assert "empty_df" in exc.value.code


@patch("aliexpress.datareader.pd.read_excel")
def test_read_groups_excel_missing_col(mock_read_excel):
    df = pd.DataFrame(
        {
            "Groepen": ["De Flamingo's"],
            "Jongens": [np.nan],
            "Meisjes": [6],
        }
    )

    mock_read_excel.return_value = df
    with pytest.raises(errors.ValidationError) as exc:
        datareader.read_groups_excel("groups.xlsx")
    assert "empty_mandatory_columns_groups_to" == exc.value.code
