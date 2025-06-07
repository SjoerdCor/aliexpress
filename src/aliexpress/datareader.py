"""Read and transform the input sheet to a workable DataFrame"""

import math
from typing import Iterable
import warnings

import pandas as pd

from .errors import ValidationError


def validate_columns(df: pd.DataFrame, expected_columns, file_type: str) -> None:
    """Validates whether df has expected columns

    file-type is in {"preferences", "groups_to", "not_together"}

    Raises
    ------
    Validation error if not matching
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
    # TODO: Make this more readable
    if mask == "Gewicht":
        mask = df["Gewicht"] < 0
    elif mask == "Liever niet met":
        mask = df["TypeWens"] == "Liever niet met"
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
        return x.strip().title().replace(" ", "")
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

        df = df.iloc[3:]

        self._validate_input(df)

        return df

    def clean_input(self, df):
        df.index = df.index.map(clean_name)

        # Clean each column
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(clean_name)
        return df

    def _validate_input(self, df):
        expected_columns = pd.MultiIndex.from_tuples(
            [
                ("MinimaleTevredenheid", pd.NA, pd.NA),
                ("Jongen/meisje", pd.NA, pd.NA),
                ("Stamgroep", pd.NA, pd.NA),
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
            ],
            names=["TypeWens", "Nr", "TypeWaarde"],
        )
        validate_columns(df, expected_columns, "preferences")

        duplicated = df.index[df.index.duplicated()].unique().tolist()
        if duplicated:
            raise ValidationError(
                code="duplicate_students_preferences",
                context={"duplicated": ",".join(duplicated)},
                technical_message="Non-unique leerlingen detected in input data.",
            )

        incorrect = ~df["Jongen/meisje"].isin(["Jongen", "Meisje"]).squeeze()
        if incorrect.any():
            raise ValidationError(
                code="wrong_sex",
                context={
                    "students_incorrect_sex": ",".join(
                        incorrect[incorrect].index.tolist()
                    )
                },
                technical_message=f"Wrong or unknown geslacht for {incorrect[incorrect].index.tolist()}",
            )

    def restructure(self) -> None:
        """Restructures voorkeuren DataFrame from wide to long format with default values."""
        self.df = (
            self.df.drop(
                columns=self.df.columns[
                    self.df.columns.get_level_values(0).isin(self.student_info_cols)
                ]
            )
            .stack(["TypeWens", "Nr"], future_stack=True)
            .dropna(subset="Waarde")
            .fillna({"Gewicht": 1})
        )

    def validate_preferences(self, all_to_groups=None) -> None:
        """Validates voorkeuren DataFrame structure and values."""
        expected_index_names = ["Leerling", "TypeWens", "Nr"]
        if self.df.index.names != expected_index_names:
            raise ValidationError(
                code="wrong_index_names_preferences",
                technical_message=f"Invalid index names. Expected {expected_index_names}.",
            )
        expected = {"Gewicht", "Waarde"}
        if set(self.df.columns) != expected:
            raise ValidationError(
                code="wrong_columns_preferences",
                technical_message=f"Invalid columns! Expected {sorted(expected)}, got {sorted(self.df.columns)}",
            )

        if (self.df["Gewicht"] <= 0).any():
            raise ValidationError(
                code="negative_weights_preferences",
                technical_message="All 'Gewicht' values must be positive.",
            )

        all_leerlingen = self.input.index.get_level_values("Leerling").unique().tolist()
        accepted_values = {
            "Niet in": all_to_groups or [],
            "Graag met": all_leerlingen + (all_to_groups or []),
            "Liever niet met": all_leerlingen + (all_to_groups or []),
        }

        for wishtype, allowed_values in accepted_values.items():
            try:
                # The default value in the lambda prevents pylint cell-var-from-loop
                invalid_values = self.df.xs(wishtype, level="TypeWens")["Waarde"].loc[
                    lambda x, allowed_values=allowed_values: ~x.isin(allowed_values)
                ]
                if not invalid_values.empty:
                    raise ValidationError(
                        "invalid_values_preferences",
                        context={
                            "wishtype": wishtype,
                            "invalid_values": ",".join(invalid_values.tolist()),
                        },
                        technical_message=f"Invalid values in '{wishtype}':\n{invalid_values}",
                    )
            except KeyError:
                warnings.warn(f"No entries found for wish type '{wishtype}'")

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
    expected_cols = [
        "Max aantal samen",
        "Leerling 1",
        "Leerling 2",
        "Leerling 3",
        "Leerling 4",
        "Leerling 5",
        "Leerling 6",
        "Leerling 7",
        "Leerling 8",
        "Leerling 9",
        "Leerling 10",
        "Leerling 11",
        "Leerling 12",
    ]
    validate_columns(df_not_together, expected_cols, "not_together")
    result = []

    def _validate(group, max_aantal_samen, students, n_groups):
        # TODO: validate group not empty
        duplicated = group.duplicated()
        if duplicated.any():
            raise ValidationError(
                code="duplicated_students_not_together",
                context={
                    "row": i + 1,
                    "duplicated_students": ",".join(group[duplicated].tolist()),
                },
                technical_message=f"Duplicated students on row {i + 1}:\n {group[duplicated]}",
            )

        known_students = group.isin(students)
        if not known_students.all():
            raise ValidationError(
                code="unknown_students_not_together",
                context={
                    "row": i + 1,
                    "unknow_students": ",".join(group[~known_students]),
                },
                technical_message=f"Unknown students on row {i + 1}:\n{group[~known_students]}",
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

    for i, row in df_not_together.iterrows():
        group = row.filter(like="Leerling").dropna().apply(clean_name)
        max_aantal_samen = row["Max aantal samen"]

        _validate(group, max_aantal_samen, students, n_groups)

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
    expected_columns = ["Groepen", "Jongens", "Meisjes"]
    validate_columns(df, expected_columns, "groups_to")
    return df.set_index("Groepen").to_dict(orient="index")
