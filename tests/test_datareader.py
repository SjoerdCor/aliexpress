# pylint: disable=redefined-outer-name # for fixtures
# pylint: disable=protected-access

"""Tests for the datareader module in the aliexpress package"""

from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from aliexpress import datareader, errors


@pytest.fixture
def valid_voorkeuren_df():
    """Fixture for a valid preferences DataFrame with the expected structure."""
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


def test_validate_columns_success():
    """Test that validate_columns does not raise an error for correct columns."""
    df = pd.DataFrame(columns=["A", "B", "C"])
    datareader.validate_columns(df, ["A", "B", "C"], "test")


def test_validate_columns_extra_and_missing():
    """Test that validate_columns raises an error for extra and missing columns."""
    df = pd.DataFrame(columns=["A", "B", "D"])
    with pytest.raises(errors.ValidationError) as exc:
        datareader.validate_columns(df, ["A", "B", "C"], "test")
    expected = "Wrong columns for test: \nmissing={'C'}\nextra={'D'}"
    assert str(exc.value) == expected


def test_toggle_negative_weights():
    """Test that toggle_negative_weights correctly toggles weights and TypeWens."""
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
    """Test that toggle_negative_weights correctly toggles weights for 'Liever niet met' mask."""
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
        ("<script>", "Script"),  # HTML-unsafe chars stripped
        ("ANNa-MAriE", "Anna-Marie"),
        ("Anne marie", "AnneMarie"),  # spaces removed, case folded
        (42, 42),  # not a string
    ],
)
def test_matching_key(input_str, expected):
    """matching_key strips HTML-unsafe chars and spaces, and folds case."""
    assert datareader.matching_key(input_str) == expected


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("  John  ", "John"),  # only edge whitespace trimmed
        ("Anne Claire", "Anne Claire"),  # spaces preserved
        ("van der Berg", "van der Berg"),  # capitals/spaces preserved
        ("McDonald", "McDonald"),
        ("O'Brien", "O'Brien"),  # apostrophe kept
        (42, 42),  # not a string
    ],
)
def test_display_name(input_str, expected):
    """display_name keeps the name as entered, only trimming edge whitespace."""
    assert datareader.display_name(input_str) == expected


def test_matching_key_collapses_case_and_space():
    """A wish and a student spelled with different case/spacing match on the same key."""
    assert datareader.matching_key("anne claire") == datareader.matching_key(
        "Anne Claire"
    )


@patch("aliexpress.datareader.pd.read_excel")
def test_voorkeuren_processor_init(mock_read_excel, valid_voorkeuren_df):
    """Test that VoorkeurenProcessor initializes correctly with a valid DataFrame."""
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
    processor = datareader.VoorkeurenProcessor("dummy.xlsx")
    assert isinstance(processor.df, pd.DataFrame)
    assert processor.df.equals(processor.input)
    pd.testing.assert_frame_equal(processor.df, expected)


def test_voorkeuren_processor_wrong_columns(valid_voorkeuren_df):
    """Test that VoorkeurenProcessor raises an error for wrong columns."""
    df = valid_voorkeuren_df.copy()
    df = df.iloc[:, :-1]
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    with pytest.raises(errors.ValidationError) as exc:
        processor._validate_input(df.iloc[:, :-1])
    assert "wrong_columns_preferences" in exc.value.code


def test_voorkeuren_processor_empty_df(valid_voorkeuren_df):
    """Test that VoorkeurenProcessor raises an error for an empty DataFrame."""
    df = valid_voorkeuren_df.copy()
    df = df.iloc[:0, :]
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    with pytest.raises(pa.errors.SchemaError) as excinfo:
        processor._validate_input(df)
    exc = excinfo.value
    assert exc.reason_code == pa.errors.SchemaErrorReason.DATAFRAME_CHECK
    assert exc.check.name == "empty_df"
    assert exc.filetype == "voorkeuren"


def test_voorkeuren_processor_no_preferences(valid_voorkeuren_df):
    """Test that VoorkeurenProcessor returns an empty DataFrame when no preferences are provided."""
    df = valid_voorkeuren_df.copy().iloc[1:]
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    processor.df = df
    processor.input = df
    df_processed = processor.process(["Oranje", "Blauw"])
    expected_index = pd.MultiIndex.from_arrays(
        [
            np.array([], dtype=object),
            np.array([], dtype=object),
            np.array([], dtype=int),
        ],  # Nr
        names=["Leerling", "TypeWens", "Nr"],
    )
    expected = pd.DataFrame(
        columns=pd.Index(["Waarde", "Gewicht"], name="TypeWaarde"),
        index=expected_index,
    ).astype({"Gewicht": "float64"})
    pd.testing.assert_frame_equal(df_processed, expected)


def test_voorkeuren_processor_mandatory_columns(valid_voorkeuren_df):
    """Test that VoorkeurenProcessor raises an error for missing mandatory columns."""
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)

    df = valid_voorkeuren_df.copy()
    df["Stamgroep"] = np.nan
    with pytest.raises(pa.errors.SchemaError) as exc:
        processor._validate_input(df)
        assert exc.reason_code == pa.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS
        assert "Stamgroep" in exc.failure_cases
        assert exc.filetype == "voorkeuren"

    df = valid_voorkeuren_df.copy()
    df["Jongen/meisje"] = np.nan
    with pytest.raises(pa.errors.SchemaError) as excinfo:
        processor._validate_input(df)

    exc = excinfo.value
    assert exc.reason_code == pa.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS
    assert ("Jongen/meisje", np.nan, np.nan) == exc.column_name
    assert exc.filetype == "voorkeuren"


def test_voorkeuren_processor_wrong_datatype(valid_voorkeuren_df):
    """Test that VoorkeurenProcessor raises an error for wrong/inconvertible datatype"""
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)

    df = valid_voorkeuren_df.copy()
    df.loc["John", ("MinimaleTevredenheid", np.nan, np.nan)] = "String"
    with pytest.raises(pa.errors.SchemaError) as excinfo:
        processor._validate_input(df)

    exc = excinfo.value
    assert exc.reason_code == pa.errors.SchemaErrorReason.DATATYPE_COERCION
    assert ("MinimaleTevredenheid", np.nan, np.nan) == exc.schema.name
    assert exc.filetype == "voorkeuren"

    df = valid_voorkeuren_df.copy()
    df.loc["John", ("Liever niet met", 1.0, "Gewicht")] = "String"
    with pytest.raises(pa.errors.SchemaError) as excinfo:
        processor._validate_input(df)
    exc = excinfo.value
    assert exc.reason_code == pa.errors.SchemaErrorReason.DATATYPE_COERCION
    assert "Gewicht" in exc.schema.name
    assert exc.filetype == "voorkeuren"


def test_voorkeuren_processor_clean_input():
    """Test that VoorkeurenProcessor cleans input DataFrame correctly."""
    df = pd.DataFrame(
        {("A", "B", "C"): ["  john ", "<script>"]}, index=[" alice ", "bob"]
    )
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    cleaned = processor.clean_input(df)
    assert "John" in cleaned.iloc[:, 0].values
    assert "Script" in cleaned.iloc[:, 0].values
    assert "Alice" in cleaned.index


def test_voorkeuren_processor_process(valid_voorkeuren_df):
    """Test that VoorkeurenProcessor processes preferences correctly."""
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
    """ "Test that VoorkeurenProcessor raises an error for duplicate student preferences."""
    df = pd.concat([valid_voorkeuren_df, valid_voorkeuren_df])
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    with pytest.raises(pa.errors.SchemaError) as exc:
        processor._validate_input(df)
        assert exc.reason_code == pa.errors.SchemaErrorReason.SERIES_CONTAINS_DUPLICATES
        assert "Leerling" in exc.failure_cases
        assert exc.filetype == "voorkeuren"


def test_voorkeuren_processor_validate_input_wrong_sex(valid_voorkeuren_df):
    """Test that VoorkeurenProcessor raises an error for sex that is not Jongen or Meisje."""
    df = valid_voorkeuren_df.copy()
    df.iloc[0, df.columns.get_loc(("Jongen/meisje", np.nan, np.nan))] = "Alien"
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    with pytest.raises(pa.errors.SchemaError) as exc:

        processor._validate_input(df)
        assert exc.reason_code == pa.errors.SchemaErrorReason.DATAFRAME_CHECK
        assert exc.column_name == ("Jongen/meisje", np.nan, np.nan)
        assert exc.filetype == "voorkeuren"


def test_voorkeuren_processor_validate_input_duplicated_values(valid_voorkeuren_df):
    """ "Test that VoorkeurenProcessor raises an error for duplicated values in preferences."""
    df = valid_voorkeuren_df.copy()
    df.loc["John", ("Graag met", 1, "Waarde")] = "Jane"
    df.loc["John", ("Graag met", 2, "Waarde")] = "Jane"

    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    processor.input = df
    processor.df = df.copy()
    processor.restructure()
    with pytest.raises(pa.errors.SchemaError) as excinfo:
        processor.validate_preferences(["Oranje", "Blauw"])
    exc = excinfo.value
    assert exc.reason_code == pa.errors.SchemaErrorReason.DATAFRAME_CHECK
    assert exc.check.name == "duplicated_values_preferences"
    assert exc.filetype == "voorkeuren"


def test_voorkeuren_processor_negative_gewicht(valid_voorkeuren_df):
    """Test that VoorkeurenProcessor raises on negative gewicht."""
    df = valid_voorkeuren_df.copy()
    df.loc["John", ("Graag met", 1, "Gewicht")] = -1

    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    processor.input = df
    processor.df = df.copy()
    processor.restructure()

    with pytest.raises(pa.errors.SchemaError) as excinfo:
        processor.validate_preferences(["Oranje", "Blauw"])
    err = excinfo.value
    assert err.reason_code == pa.errors.SchemaErrorReason.DATAFRAME_CHECK
    assert err.check.name == "greater_than" and "Gewicht" in err.column_name
    assert err.filetype == "voorkeuren"


def test_voorkeuren_processor_validate_preferences_invalid_values(valid_voorkeuren_df):
    """Test that VoorkeurenProcessor raises an error for unknown leerling/group."""
    df = valid_voorkeuren_df.copy()

    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    processor.input = df
    processor.df = df.copy()
    processor.restructure()

    with pytest.raises(pa.errors.SchemaError) as excinfo:
        processor.validate_preferences(["Blauw"])
    err = excinfo.value
    assert err.reason_code == pa.errors.SchemaErrorReason.DATAFRAME_CHECK
    assert err.check.name == "invalid_values_preferences"


def test_voorkeuren_processor_weight_missing_name(valid_voorkeuren_df):
    """Test that VoorkeurenProcessor raises an error for missing name in weight column."""
    df = valid_voorkeuren_df.copy()
    df.loc["John", ("Graag met", 1, "Waarde")] = np.nan
    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    processor.df = df
    processor.input = df
    processor.restructure()

    with pytest.raises(pa.errors.SchemaError) as exc:
        processor.validate_preferences(["Blauw", "Oranje"])
    assert exc.value.reason_code == pa.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS


def test_voorkeuren_processor_process_and_get_students_meta_info(valid_voorkeuren_df):
    """Test that VoorkeurenProcessor retrieves student meta info correctly."""
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


def test_validate_not_together_success():
    """Valid rules pass without error and are returned unchanged."""
    rules = [{"group": {"Alice", "Bob"}, "Max_aantal_samen": 1}]
    assert (
        datareader.validate_not_together(rules, ["Alice", "Bob"], n_groups=2) == rules
    )


def test_validate_not_together_empty():
    """Empty rule list is always valid."""
    # pylint: disable-next=use-implicit-booleaness-not-comparison  # exact return type matters
    assert datareader.validate_not_together([], ["Alice", "Bob"], n_groups=2) == []


def test_validate_not_together_too_few_students():
    """A rule with fewer than 2 students raises an error."""
    rules = [{"group": {"Alice"}, "Max_aantal_samen": 1}]
    with pytest.raises(errors.ValidationError) as exc:
        datareader.validate_not_together(rules, ["Alice", "Bob"], n_groups=2)
    assert exc.value.code == "too_few_students_not_together"


def test_validate_not_together_unknown_student():
    """A student not in the known list raises an error."""
    rules = [{"group": {"Alice", "Unknown"}, "Max_aantal_samen": 1}]
    with pytest.raises(errors.ValidationError) as exc:
        datareader.validate_not_together(rules, ["Alice", "Bob"], n_groups=2)
    assert exc.value.code == "unknown_student_not_together"
    assert "Unknown" in exc.value.context["unknown_students"]


def test_validate_not_together_too_strict():
    """A rule that cannot be distributed over n_groups raises an error."""
    rules = [{"group": {"Alice", "Bob", "Charlie"}, "Max_aantal_samen": 1}]
    with pytest.raises(errors.ValidationError) as exc:
        datareader.validate_not_together(rules, ["Alice", "Bob", "Charlie"], n_groups=2)
    assert exc.value.code == "too_strict_not_together"


def test_validate_not_together_invalid_max_samen():
    """max_samen < 1 raises an error."""
    rules = [{"group": {"Alice", "Bob"}, "Max_aantal_samen": 0}]
    with pytest.raises(errors.ValidationError) as exc:
        datareader.validate_not_together(rules, ["Alice", "Bob"], n_groups=2)
    assert exc.value.code == "invalid_max_samen_not_together"


@patch("aliexpress.datareader.pd.read_excel")
def test_read_groups_excel_success(mock_read_excel):
    """Test that read_groups_excel reads a DataFrame with groups correctly."""
    df = pd.DataFrame(
        {
            "Groepen": ["De Flamingo's"],
            "Jongens": [5],
            "Meisjes": [6],
        }
    )
    mock_read_excel.return_value = df
    groups_to, group_display = datareader.read_groups_excel("groups.xlsx")
    assert groups_to == {"DeFlamingos": {"Jongens": 5, "Meisjes": 6}}
    assert group_display == {"DeFlamingos": "De Flamingo's"}


@patch("aliexpress.datareader.pd.read_excel")
def test_read_groups_excel_empty(mock_read_excel):
    """Test that read_groups_excel raises an error for an empty DataFrame."""
    mock_read_excel.return_value = pd.DataFrame(
        columns=["Groepen", "Jongens", "Meisjes"]
    )
    with pytest.raises(pa.errors.SchemaError) as exc:
        datareader.read_groups_excel("groups.xlsx")
    assert exc.value.reason_code == pa.errors.SchemaErrorReason.DATAFRAME_CHECK
    assert exc.value.check.name == "empty_df" and exc.value.filetype == "groepen"


@patch("aliexpress.datareader.pd.read_excel")
def test_read_groups_excel_missing_col(mock_read_excel):
    """Test that read_groups_excel raises an error for missing mandatory columns."""
    df = pd.DataFrame(
        {
            "Groepen": ["De Flamingo's"],
            "Jongens": [np.nan],
            "Meisjes": [6],
        }
    )

    mock_read_excel.return_value = df
    with pytest.raises(pa.errors.SchemaError) as exc:
        datareader.read_groups_excel("groups.xlsx")
    assert exc.value.reason_code == pa.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS
    assert exc.value.column_name == "Jongens"
    assert exc.value.filetype == "groepen"


@pytest.fixture
def reader_mini():
    """Read small xml"""
    xml_mini = b"""<?xml version="1.0" encoding="utf-8"?>
<EDEX>
  <groepen>
    <groep key="G1">
      <naam>Groep 3A</naam>
      <jaargroep>3</jaargroep>
    </groep>
  </groepen>
  <leerlingen>
    <leerling key="L1">
      <achternaam>Jansen</achternaam>
      <voornamen>Peter Jan</voornamen>
      <roepnaam>Peter</roepnaam>
      <voorletters>PJ</voorletters>
      <geboortedatum>2015-04-12</geboortedatum>
      <geslacht>1</geslacht>
      <jaargroep>3</jaargroep>
      <instroomdatum>2021-08-16</instroomdatum>
      <groep key="G1"/>
    </leerling>
    <leerling key="L2">
      <achternaam>De Vries</achternaam>
      <voornamen>Anna</voornamen>
      <roepnaam>Anna</roepnaam>
      <voorletters>A</voorletters>
      <geboortedatum>2015-11-03</geboortedatum>
      <geslacht>2</geslacht>
      <jaargroep>3</jaargroep>
      <instroomdatum>2021-08-16</instroomdatum>
      <groep key="G1"/>
    </leerling>
  </leerlingen>
</EDEX>
"""

    return datareader.EdexReader(BytesIO(xml_mini))


@pytest.fixture
def reader_combi():
    """Read file with leerlingen and groepen"""
    xml_combi = b"""<?xml version="1.0" encoding="utf-8"?>
<EDEX>
  <groepen>
    <groep key="G2">
      <naam>Groep 4B</naam>
      <jaargroep>4</jaargroep>
    </groep>
    <groep key="G3">
      <naam>Groep 5C</naam>
      <jaargroep>5</jaargroep>
    </groep>
    <groep key="G4_4">
      <naam>Groep 4/5 Combi</naam>
      <jaargroep>4</jaargroep>
    </groep>
    <groep key="G4_5">
      <naam>Groep 4/5 Combi</naam>
      <jaargroep>5</jaargroep>
    </groep>
  </groepen>
  <leerlingen>
    <leerling key="L3">
      <achternaam>Mulder</achternaam>
      <voornamen>Kees</voornamen>
      <roepnaam>Kees</roepnaam>
      <voorletters>K</voorletters>
      <geboortedatum>2014-07-20</geboortedatum>
      <geslacht>1</geslacht>
      <jaargroep>4</jaargroep>
      <instroomdatum>2020-08-17</instroomdatum>
      <groep key="G2"/>
    </leerling>
    <leerling key="L4">
      <achternaam>Bakker</achternaam>
      <voornamen>Sophie</voornamen>
      <roepnaam>Sophie</roepnaam>
      <voorletters>S</voorletters>
      <geboortedatum>2013-12-05</geboortedatum>
      <geslacht>2</geslacht>
      <jaargroep>5</jaargroep>
      <instroomdatum>2019-08-19</instroomdatum>
      <groep key="G4_5"/>
    </leerling>
  </leerlingen>
</EDEX>
"""

    return datareader.EdexReader(BytesIO(xml_combi))


@pytest.fixture
def reader_edge():
    """Read xml with ongespecificeerd or unknown geslacht"""
    xml_edge = b"""<?xml version="1.0" encoding="utf-8"?>
<EDEX>
  <groepen>
    <groep key="G5">
      <naam>Groep 6A</naam>
      <jaargroep>6</jaargroep>
    </groep>
  </groepen>
  <leerlingen>
    <leerling key="L5">
      <achternaam>van Dijk</achternaam>
      <voornamen>Sam</voornamen>
      <roepnaam>Sam</roepnaam>
      <voorletters>S</voorletters>
      <geboortedatum>2012-09-01</geboortedatum>
      <geslacht>0</geslacht>
      <jaargroep>6</jaargroep>
      <instroomdatum>2018-08-20</instroomdatum>
      <groep key="G5"/>
    </leerling>
    <leerling key="L6">
      <achternaam>Unknown</achternaam>
      <voornamen>Case</voornamen>
      <roepnaam>Case</roepnaam>
      <voorletters>C</voorletters>
      <geboortedatum>2012-01-15</geboortedatum>
      <geslacht>9</geslacht>
      <jaargroep>6</jaargroep>
      <instroomdatum>2018-08-20</instroomdatum>
      <groep key="G5"/>
    </leerling>
  </leerlingen>
</EDEX>
"""
    return datareader.EdexReader(BytesIO(xml_edge))


def test_parse_leerlingen_mini(reader_mini):
    """Test small reads leerlingen fine"""
    df = reader_mini.df_leerlingen
    assert df.shape[0] == 2
    assert "roepnaam" in df.columns
    assert df.loc["L1", "geslacht"] == "Jongen"
    assert df.loc["L2", "geslacht"] == "Meisje"
    assert df.loc["L1", "jaargroep"] == 3


def test_parse_groepen_mini(reader_mini):
    "Test groepen are parsed fine"
    df = reader_mini.df_groepen
    assert df.shape[0] == 1
    assert df.loc["G1", "jaargroep"] == 3
    assert df.loc["G1", "naam"] == "Groep 3A"


def test_get_full_df_mini(reader_mini):
    "Test combination of small file is fine"
    df = reader_mini.get_full_df()
    assert "groepsnaam" in df.columns
    assert df.loc["L1", "groepsnaam"] == "Groep 3A"
    assert "jaargroep_groep" not in df.columns


def test_combi_group_names_and_years(reader_combi):
    """Test combigroepen works fine"""
    df = reader_combi.df_groepen
    # er bestaan twee groep-keys met dezelfde 'naam' (combi)
    combi_rows = df[df["naam"] == "Groep 4/5 Combi"]
    assert set(combi_rows["jaargroep"].astype(int).tolist()) == {4, 5}


def test_leerlingen_in_combigroep(reader_combi):
    """Test leerlingen in combigroep"""
    df = reader_combi.get_full_df()
    # Sophie zit in combi-groep
    assert df.loc["L4", "groepsnaam"].startswith("Groep 4/5")


def test_edge_cases(reader_edge):
    """Test edge case for geslachten"""
    df = reader_edge.df_leerlingen
    assert df.loc["L5", "geslacht"] == "Onbekend"
    assert df.loc["L6", "geslacht"] == "Niet gespecificeerd"


def test_niet_in_second_column_accepts_group_name(valid_voorkeuren_df):
    """Regression: second Niet-in column must accept a group name, not be validated as a float.

    Bug: ("Niet in", 2.0, "Waarde") was mapped to gewicht_check instead of waarde_check,
    causing a DATATYPE_COERCION error when a teacher fills in a second forbidden group.
    """
    df = valid_voorkeuren_df.copy()
    df[("Niet in", 2.0, "Waarde")] = df[("Niet in", 2.0, "Waarde")].astype(object)
    df.loc["John", ("Niet in", 2.0, "Waarde")] = "Blauw"

    processor = datareader.VoorkeurenProcessor.__new__(datareader.VoorkeurenProcessor)
    result = processor._validate_input(df)
    assert result.loc["John", ("Niet in", 2.0, "Waarde")] == "Blauw"
