"""Read and transform the input sheet to a workable DataFrame"""

import math
from typing import Iterable
import warnings

import pandas as pd


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


class ValidationError(Exception):
    def __init__(self, code, context=None, technical_message=None):
        super().__init__(technical_message or code)
        self.code = code
        self.context = context or {}
        self.technical_message = technical_message


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

        duplicated = df.index[df.index.duplicated()].unique().tolist()
        if duplicated:
            # \n is not allowed in f-strings
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
        if self.df.index.names != ["Leerling", "TypeWens", "Nr"]:
            raise ValueError(
                "Invalid index names. Expected ['Leerling', 'TypeWens', 'Nr']."
            )
        expected = {"Gewicht", "Waarde"}
        if set(self.df.columns) != expected:
            raise ValueError(
                f"Invalid columns! Expected {sorted(expected)}, got {sorted(self.df.columns)}"
            )

        if (self.df["Gewicht"] <= 0).any():
            raise ValueError("All 'Gewicht' values must be positive.")

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
                    raise ValueError(
                        f"Invalid values in '{wishtype}':\n{invalid_values}"
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
    result = []

    def _validate(group, max_aantal_samen, students, n_groups):
        duplicated = group.duplicated()
        if duplicated.any():
            raise ValueError(
                f"Duplicated students on row {i + 1}:\n {group[duplicated]}"
            )

        known_students = group.isin(students)
        if not known_students.all():
            raise ValueError(
                f"Unknown students on row {i + 1}:\n{group[~known_students]}"
            )

        if len(group) / max_aantal_samen > n_groups:
            msg = (
                f"Cannot divide {len(group)} students over {n_groups} groups in "
                f"subgroups of max size {max_aantal_samen}. Must be at least "
                f"{math.ceil(len(group) / n_groups)} (row {i + 1})"
            )
            raise ValueError(msg)

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
