"""Read and transform the input sheet to a workable DataFrame"""

import math
import re
import warnings
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


def check_mandatory_columns(
    df: pd.DataFrame, mandatory_columns: list, file_type: str, check_index=True
):
    """Check whether all mandatory columns are filled

    file-type is in {"preferences", "groups_to", "not_together"}

    Raises
    ------
    ValidationError if mandatory columns contain NA

    """
    failed_columns = []
    if check_index:
        if df.index.isna().any():
            if isinstance(df.index, pd.MultiIndex):
                parts = [str(c) for c in df.index.names if pd.notna(c)]
                name = "_".join(parts)
            else:
                name = df.index.name
            failed_columns.append(name)
    illegal_cols = df[mandatory_columns].isna().any().loc[lambda s: s]
    for col in illegal_cols.index.tolist():
        if isinstance(col, tuple):
            col_name = "_".join(str(c) for c in col if pd.notna(c))
        else:
            col_name = str(col)
        failed_columns.append(col_name)

    if failed_columns:
        raise ValidationError(
            code=f"empty_mandatory_columns_{file_type}",
            context={"failed_columns": ", ".join(failed_columns)},
            technical_message=(
                "Mandatory columns not filled:\n"
                f" index: {df.index.isna().sum()},\n"
                f"{df[mandatory_columns].isna().sum()}"
            ),
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
        all_leerlingen = self.input.index.get_level_values("Leerling").tolist()

        schema = pa.DataFrameSchema(
            columns={
                "Waarde": pa.Column(str),
                "Gewicht": pa.Column(float),
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
                    pa.Index(
                        int,
                        name="Nr",
                        checks=pa.Check.greater_than(0, name="positive_gewicht"),
                    ),
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
    )
    try:
        df_not_together = schema.validate(df_not_together)
    except pa.errors.SchemaError as exc:
        if exc.reason_code == pa.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS:
            raise ValidationError(
                code="empty_mandatory_columns_not_together",
                context={"failed_columns": exc.failure_cases},
                technical_message=(
                    f"Mandatory columns not filled:\n {exc.failure_cases}"
                ),
            ) from exc
        if exc.reason_code == pa.errors.SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME:
            raise ValidationError(
                "wrong_columns_not_together",
                context={"wrong_columns": "Ontbrekende kolommen: " + exc.failure_cases},
                technical_message=f"Wrong columns for not_together: {exc.failure_cases}",
            ) from exc
        if exc.reason_code == pa.errors.SchemaErrorReason.COLUMN_NOT_IN_SCHEMA:
            raise ValidationError(
                "wrong_columns_not_together",
                context={"wrong_columns": "Extra kolommen: " + exc.failure_cases},
                technical_message=f"Wrong columns for not_together: {exc.failure_cases}",
            ) from exc
        if exc.reason_code == pa.errors.SchemaErrorReason.DATATYPE_COERCION:
            raise ValidationError(
                code="wrong_datatype",
                context={
                    "failed_columns": exc.schema.name,
                    "filetype": "niet_samen",
                },
                technical_message=(
                    f"Column {exc.schema.name} can not be converted to the correct datatype\n"
                    f"{exc.failure_cases}"
                ),
            ) from exc
        if (
            exc.reason_code == pa.errors.SchemaErrorReason.DATAFRAME_CHECK
            and exc.check.name == "isin"
        ):
            raise ValidationError(
                code="unknown_students_not_together",
                context={
                    "row": exc.failure_cases,
                    "unknown_students": ", ".join(
                        exc.failure_cases["failure_case"].astype(str)
                    ),
                },
                technical_message=f"Unknown students: {exc.failure_cases}",
            ) from exc
        raise exc

    def _validate(group, max_aantal_samen, n_groups):
        """Perform row-wise checks"""
        duplicated = group.duplicated()
        if duplicated.any():
            raise ValidationError(
                code="duplicated_students_not_together",
                context={
                    "row": i + 1,
                    "duplicated_students": ",".join(
                        group[duplicated].astype(str).tolist()
                    ),
                },
                technical_message=f"Duplicated students on row {i + 1}:\n {group[duplicated]}",
            )

        if len(group) / max_aantal_samen > n_groups:
            msg = (
                f"Cannot divide {len(group)} students over {n_groups} groups in "
                f"subgroups of max size {max_aantal_samen}. (row {i + 1})"
            )
            raise ValidationError(
                "too_strict_not_together",
                context={
                    "n_students": len(group),
                    "n_groups": n_groups,
                    "max_aantal_samen": max_aantal_samen,
                    "acceptabel_max_samen": math.ceil(len(group) / n_groups),
                    "row": i + 1,
                },
                technical_message=msg,
            )

    result = []
    for i, row in df_not_together.iterrows():
        group = row.filter(like="Leerling").dropna().apply(clean_name)
        max_aantal_samen = row["Max aantal samen"]

        _validate(group, max_aantal_samen, n_groups)

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
