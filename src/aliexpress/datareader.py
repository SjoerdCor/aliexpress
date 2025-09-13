"""Read and transform the input sheet to a workable DataFrame"""

import re
import warnings
import xml.etree.ElementTree as ET
from typing import Iterable

import numpy as np
import pandas as pd
import pandera.pandas as pa

from .errors import ValidationError


def validate_schema_with_filetype(
    df: pd.DataFrame, schema: pa.DataFrameSchema, filetype: str
) -> pd.DataFrame:
    """Validates a DataFrame against a given schema and raises a SchemaError
    with filetype context if validation fails.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    schema : pa.DataFrameSchema
        The pandera DataFrameSchema to validate against.
    filetype : str
        The type of file being validated, used in error messages.

    Returns
    -------
    pd.DataFrame
        The validated DataFrame.
    """
    try:
        df = schema.validate(df)
    except pa.errors.SchemaError as exc:
        exc.filetype = filetype  # Attach filetype to the exception for context
        raise exc
    return df


def create_check_empty_df():
    """Creates a pandera Check to ensure DataFrame is not empty."""
    return pa.Check(
        lambda df: len(df) > 0,
        name="empty_df",
        error="DataFrame cannot be empty",
    )


def validate_columns(df: pd.DataFrame, expected_columns, file_type: str) -> None:
    """Validates whether df has expected columns

    file-type is in {"preferences", "groups_to", "not_together"}

    Raises
    ------
    ValidationError if not matching
    """

    def flatten_column(col: tuple) -> str:
        "Comparable by removing nan, readable: tuple -> str"
        parts = [str(c) for c in col if pd.notna(c)]
        return "_".join(parts)

    if isinstance(df.columns, pd.MultiIndex):
        actual = {flatten_column(c) for c in df.columns}
        expected = {flatten_column(c) for c in expected_columns}
    else:
        actual = set(df.columns)
        expected = set(expected_columns)

    missing = expected - actual
    extra = actual - expected

    if missing or extra:
        msg = ""
        if missing:
            msg += f"Ontbrekende kolommen: {', '.join(missing)}. \n"
        if extra:
            msg += f"Onverwachte kolommen: {', '.join(extra)}."
        raise ValidationError(
            f"wrong_columns_{file_type}",
            context={"wrong_columns": msg},
            technical_message=f"Wrong columns for {file_type}: \n{missing=}\n{extra=}",
        )


def toggle_negative_weights(df: pd.DataFrame, mask="Gewicht") -> pd.DataFrame:
    """Adjusts 'Liever niet met'/'Graag met' category by negating weight and renaming.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing preferences in long-form, with right indexes
        annd Gewicht as column

    Returns
    -------
        pd.DataFrame
            Of the same shape, but with negated Gewicht and TypeWens
    """
    df = df.reset_index()
    if mask == "Gewicht":
        mask = df["Gewicht"] < 0
    elif mask == "Liever niet met":
        mask = df["TypeWens"] == "Liever niet met"
    else:
        raise ValueError(
            "mask should be either 'Gewicht' or 'Liever niet met', "
            f"got {mask} instead."
        )
    df.loc[mask, "Gewicht"] = -df["Gewicht"]
    df.loc[mask, "TypeWens"] = df.loc[mask, "TypeWens"].map(
        {"Graag met": "Liever niet met", "Liever niet met": "Graag met"}
    )

    df["Nr"] = df.groupby(["Leerling", "TypeWens"]).cumcount() + 1
    df = df.set_index(["Leerling", "TypeWens", "Nr"])
    return df


def clean_name(x):
    """Clean spaces and capitals in names"""
    if isinstance(x, str):
        html_safe = re.sub(r"[<>&\"'`=/\\]", "", x)
        return html_safe.strip().title().replace(" ", "")
    return x


class VoorkeurenProcessor:
    """Read and transform the input sheet to a workable DataFrame"""

    student_info_cols = ["MinimaleTevredenheid", "Jongen/meisje", "Stamgroep"]

    def __init__(self, filename: str = "voorkeuren.xlsx"):
        self.filename = filename
        self.input = self._read_voorkeuren().pipe(self.clean_input)
        self.df = self.input.copy()

    def _read_voorkeuren(self) -> pd.DataFrame:
        """Reads and processes the voorkeuren file into a structured DataFrame."""
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            # The data validation in the input sheet gives a UserWarning
            df = pd.read_excel(self.filename, header=None, index_col=0).rename_axis(
                "Leerling"
            )

        with pd.option_context("future.no_silent_downcasting", True):
            df.iloc[0] = df.iloc[0].ffill().infer_objects(copy=False)
            df.iloc[1] = df.iloc[1].ffill().infer_objects(copy=False)
        df.iloc[2] = df.iloc[2].replace(
            {"Naam (leerling of stamgroep)": "Waarde", "Stamgroep": "Waarde"},
        )
        df.columns = pd.MultiIndex.from_arrays(
            [df.iloc[0], df.iloc[1], df.iloc[2]], names=["TypeWens", "Nr", "TypeWaarde"]
        )

        df = df.iloc[3:].pipe(self._validate_input)
        return df

    def clean_input(self, df):
        """Cleans strings of all columns and index in the DataFrame if possible."""
        df.index = df.index.map(clean_name)

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(clean_name)
        return df

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        # This "coerce" in pandera is a bit ugly, not separating concerns
        # But it does work very easily
        waarde_check = pa.Column(object, nullable=True, coerce=True)
        gewicht_check = pa.Column(
            float, checks=pa.Check.greater_than(0), nullable=True, coerce=True
        )
        schema = pa.DataFrameSchema(
            {
                ("MinimaleTevredenheid", np.nan, np.nan): pa.Column(
                    float,
                    checks=pa.Check.less_than_or_equal_to(1),
                    nullable=True,
                    coerce=True,
                ),
                ("Jongen/meisje", np.nan, np.nan): pa.Column(
                    str, checks=pa.Check.isin(["Jongen", "Meisje"]), coerce=True
                ),
                ("Stamgroep", np.nan, np.nan): pa.Column(str),
                ("Graag met", 1.0, "Waarde"): waarde_check,
                ("Graag met", 1.0, "Gewicht"): gewicht_check,
                ("Graag met", 2.0, "Waarde"): waarde_check,
                ("Graag met", 2.0, "Gewicht"): gewicht_check,
                ("Graag met", 3.0, "Waarde"): waarde_check,
                ("Graag met", 3.0, "Gewicht"): gewicht_check,
                ("Graag met", 4.0, "Waarde"): waarde_check,
                ("Graag met", 4.0, "Gewicht"): gewicht_check,
                ("Graag met", 5.0, "Waarde"): waarde_check,
                ("Graag met", 5.0, "Gewicht"): gewicht_check,
                ("Liever niet met", 1.0, "Waarde"): waarde_check,
                ("Liever niet met", 1.0, "Gewicht"): gewicht_check,
                ("Niet in", 1.0, "Waarde"): waarde_check,
                ("Niet in", 2.0, "Waarde"): gewicht_check,
            },
            index=pa.Index(pa.String, unique=True, coerce=True),
            checks=[create_check_empty_df()],
        )
        # This check does not seem to work in pandera (perhaps because
        # of np.nan in the Index)
        expected_columns = pd.MultiIndex.from_tuples(
            schema.columns.keys(),
            names=["TypeWens", "Nr", "TypeWaarde"],
        )
        validate_columns(df, expected_columns, "preferences")

        df = validate_schema_with_filetype(df, schema, filetype="voorkeuren")
        return df

    def restructure(self) -> None:
        """Restructures voorkeuren DataFrame from wide to long format with default values."""
        self.df = (
            self.df.drop(
                columns=self.df.columns[
                    self.df.columns.get_level_values(0).isin(self.student_info_cols)
                ]
            )
            .stack(["TypeWens", "Nr"], future_stack=True)
            .dropna(how="all")
            .fillna({"Gewicht": 1})
        )

    def validate_preferences(self, all_to_groups=None) -> None:
        """Validates voorkeuren DataFrame structure and values."""

        def waarde_unique_within_leerling(df: pd.DataFrame) -> bool:
            return df.groupby("Leerling")["Waarde"].apply(lambda s: s.is_unique).all()

        def waarde_matches_typewens(
            df: pd.DataFrame, all_to_groups: list, all_leerlingen: list
        ) -> bool:
            mask_nietin = df.index.get_level_values("TypeWens") == "Niet in"
            mask_other = df.index.get_level_values("TypeWens").isin(
                ["Graag met", "Liever niet met"]
            )

            valid = pd.Series(True, index=df.index)
            valid.loc[mask_nietin] = df.loc[mask_nietin, "Waarde"].isin(all_to_groups)
            valid.loc[mask_other] = df.loc[mask_other, "Waarde"].isin(
                all_to_groups + all_leerlingen
            )
            return valid

        all_to_groups = all_to_groups or []
        try:
            all_leerlingen = self.input.index.get_level_values("Leerling").tolist()
        except KeyError:
            # Make sure it does not error here yet (if index is wrong), must throw SchemaError later
            all_leerlingen = []

        schema = pa.DataFrameSchema(
            columns={
                "Waarde": pa.Column(str),
                "Gewicht": pa.Column(float, checks=pa.Check.greater_than(0)),
            },
            index=pa.MultiIndex(
                [
                    pa.Index(str, name="Leerling"),
                    pa.Index(
                        str,
                        name="TypeWens",
                        checks=pa.Check.isin(
                            ["Niet in", "Graag met", "Liever niet met"]
                        ),
                    ),
                    pa.Index(float, name="Nr"),
                ]
            ),
            checks=[
                pa.Check(
                    waarde_unique_within_leerling,
                    name="duplicated_values_preferences",
                    error="Column 'Waarde' must be unique within each Leerling.",
                ),
                pa.Check(
                    lambda df: waarde_matches_typewens(
                        df, all_to_groups, all_leerlingen
                    ),
                    name="invalid_values_preferences",
                ),
            ],
            strict=True,
            coerce=True,
        )

        validate_schema_with_filetype(self.df, schema, filetype="voorkeuren")

    def process(self, all_to_groups: list) -> pd.DataFrame:
        """Runs the full processing pipeline.

        Parameters
        ----------
        all_to_groups : list
            The groups to which students can be sent. This is necessary to validate the
            input

        """

        self.restructure()
        self.validate_preferences(all_to_groups)
        self.df = toggle_negative_weights(self.df, mask="Liever niet met")
        return self.df

    def get_students_meta_info(self) -> dict:
        """Get all meta information about each student

        This can be useful to balance new groups

        Returns
        -------
        dict
            Per student all known information
        """
        return (
            self.input[self.student_info_cols]
            .droplevel([1, 2], "columns")
            .to_dict("index")
        )


def read_not_together(filename: str, students: Iterable, n_groups: int) -> list:
    """Reads the preferences for students who should not be togeter (in large groups)"""
    df_not_together = pd.read_excel(filename)

    def create_student_column_schema(students, nullable=True):
        return pa.Column(
            object,
            checks=pa.Check.isin(students),
            nullable=nullable,
            coerce=True,
        )

    students = list(students)  # pandera needs something thats picklable

    def no_duplicated_students(df: pd.DataFrame) -> pd.Series:
        "A row is valid if there are no duplicated students in it"
        groups = df.filter(like="Leerling").map(clean_name)
        return ~groups.apply(lambda row: row.dropna().duplicated().any(), axis=1)

    def can_be_divided(df: pd.DataFrame, n_groups: int) -> pd.Series:
        """A row is valid if the students in it can be divided over n_groups"""
        group_sizes = df.filter(like="Leerling").count("columns")
        max_samen = df["Max aantal samen"]
        return (group_sizes / max_samen) <= n_groups

    schema = pa.DataFrameSchema(
        {
            "Max aantal samen": pa.Column(
                int,
                checks=pa.Check.greater_than_or_equal_to(1),
                coerce=True,
            ),
            "Leerling 1": create_student_column_schema(students, nullable=False),
            "Leerling 2": create_student_column_schema(students, nullable=False),
            "Leerling 3": create_student_column_schema(students, nullable=True),
            "Leerling 4": create_student_column_schema(students, nullable=True),
            "Leerling 5": create_student_column_schema(students, nullable=True),
            "Leerling 6": create_student_column_schema(students, nullable=True),
            "Leerling 7": create_student_column_schema(students, nullable=True),
            "Leerling 8": create_student_column_schema(students, nullable=True),
            "Leerling 9": create_student_column_schema(students, nullable=True),
            "Leerling 10": create_student_column_schema(students, nullable=True),
            "Leerling 11": create_student_column_schema(students, nullable=True),
            "Leerling 12": create_student_column_schema(students, nullable=True),
        },
        strict=True,
        checks=[
            pa.Check(no_duplicated_students, name="duplicated_students_not_together"),
            pa.Check(can_be_divided, name="too_strict_not_together", n_groups=n_groups),
        ],
    )
    validate_schema_with_filetype(df_not_together, schema, filetype="niet_samen")

    result = []
    for _, row in df_not_together.iterrows():
        group = row.filter(like="Leerling").dropna().apply(clean_name)
        max_aantal_samen = row["Max aantal samen"]

        result.append(
            {
                "Max_aantal_samen": max_aantal_samen,
                "group": set(group),
            }
        )
    return result


def read_groups_excel(path_groups_to) -> dict:
    """Reads the information about the groups to from excel to dict"""
    df = pd.read_excel(path_groups_to)
    schema = pa.DataFrameSchema(
        {
            "Groepen": pa.Column(object, unique=True),
            "Jongens": pa.Column(
                "Int64", pa.Check.greater_than_or_equal_to(0), coerce=True
            ),
            "Meisjes": pa.Column(
                "Int64", pa.Check.greater_than_or_equal_to(0), coerce=True
            ),
        },
        checks=[create_check_empty_df()],
        strict=True,
    )

    df = validate_schema_with_filetype(df, schema, filetype="groepen")

    return (
        df.assign(Groepen=lambda df: df["Groepen"].apply(clean_name))
        .set_index("Groepen")
        .to_dict(orient="index")
    )


def get_leerlingen_from_edex(file_loc) -> pd.DataFrame:
    """Reads leerlingen from an EDEX XML file and returns them as a DataFrame."""

    tree = ET.parse(file_loc)
    root = tree.getroot()

    rows = []
    for ll in root.findall("./leerlingen/leerling"):
        data = {}
        data["key"] = ll.attrib.get("key")
        for child in ll:
            if child.tag == "groep":
                data["groepscode"] = child.attrib.get("key")
            else:
                data[child.tag] = child.text
        rows.append(data)
    df = pd.DataFrame(rows).set_index("key")
    return df


def get_groepen_from_edex(file_loc) -> pd.DataFrame:
    """Reads groepen from an EDEX XML file and returns them as a DataFrame."""
    tree = ET.parse(file_loc)
    root = tree.getroot()

    rows = []
    for gg in root.findall("./groepen/groep"):
        data = {}
        data["key"] = gg.attrib.get("key")
        for child in gg:
            data[child.tag] = child.text
        rows.append(data)
    df = pd.DataFrame(rows).set_index("key")
    return df
