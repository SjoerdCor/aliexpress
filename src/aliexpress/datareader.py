"""Read and transform the input sheet to a workable DataFrame"""

import re
import warnings
import xml.etree.ElementTree as ET
from io import BytesIO
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


def display_name(x):
    """Return the name exactly as entered, with only surrounding whitespace trimmed.

    Capitals, internal spaces, apostrophes and hyphens are preserved, so names such as
    ``O'Brien``, ``van der Berg`` and ``McDonald`` stay intact. This is the form shown to
    the user; HTML-safety of displayed names is handled at render time (the report tables
    escape their cells), not by mangling the name here.
    """
    if isinstance(x, str):
        return x.strip()
    return x


def matching_key(x):
    """Normalize a name to the internal key that matches wishes to students/groups.

    Strips HTML/URL-unsafe characters, removes spaces and folds case, so that e.g.
    ``Anne claire`` and ``Anne Claire`` collapse to the same key. The key feeds pulp
    variable names and dict lookups (and is the only form rendered raw in the sociogram),
    so keeping it free of dangerous characters is a cheap, internal safety guarantee.
    """
    if isinstance(x, str):
        html_safe = re.sub(r"[<>&\"'`=/\\]", "", x)
        return html_safe.strip().casefold().replace(" ", "")
    return x


def to_html_id(name: str) -> str:
    """Convert a name to a valid HTML element ID, safe against XSS injection.

    Replaces spaces and characters that are special in HTML/URLs with underscores.
    Use this for form field names and element IDs in the web UI — not for
    matching against solver names (use matching_key for that).
    """
    return re.sub(r"[-<>&\"'`=/\\ ]", "_", str(name).strip())


# ---------------------------------------------------------------------------
# Voorkeuren column schema — single source of truth
#
# All downstream representations (Excel header rows, data-validation column
# letters in the fill-in template) are derived from this schema's column
# keys.  Edit here to change every consumer at once.
# ---------------------------------------------------------------------------

_waarde_check = pa.Column(object, nullable=True, coerce=True)
_gewicht_check = pa.Column(
    float, checks=pa.Check.greater_than(0), nullable=True, coerce=True
)

VOORKEUREN_SCHEMA = pa.DataFrameSchema(
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
        ("Graag met", 1.0, "Waarde"): _waarde_check,
        ("Graag met", 1.0, "Gewicht"): _gewicht_check,
        ("Graag met", 2.0, "Waarde"): _waarde_check,
        ("Graag met", 2.0, "Gewicht"): _gewicht_check,
        ("Graag met", 3.0, "Waarde"): _waarde_check,
        ("Graag met", 3.0, "Gewicht"): _gewicht_check,
        ("Graag met", 4.0, "Waarde"): _waarde_check,
        ("Graag met", 4.0, "Gewicht"): _gewicht_check,
        ("Graag met", 5.0, "Waarde"): _waarde_check,
        ("Graag met", 5.0, "Gewicht"): _gewicht_check,
        ("Liever niet met", 1.0, "Waarde"): _waarde_check,
        ("Liever niet met", 1.0, "Gewicht"): _gewicht_check,
        ("Niet in", 1.0, "Waarde"): _waarde_check,
        ("Niet in", 2.0, "Waarde"): _waarde_check,
    },
    index=pa.Index(pa.String, unique=True, coerce=True),
    checks=[create_check_empty_df()],
)


class VoorkeurenProcessor:
    """Read and transform the input sheet to a workable DataFrame"""

    student_info_cols = ["MinimaleTevredenheid", "Jongen/meisje", "Stamgroep"]

    def __init__(self, filename: str = "voorkeuren.xlsx"):
        self.filename = filename
        # Filled by clean_input: matching_key -> name as entered, for the report layer.
        self.student_display: dict = {}
        self.stamgroep_display: dict = {}
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

    @staticmethod
    def _display_map(values) -> dict:
        """Map each name's matching key back to the name as entered."""
        return {matching_key(v): display_name(v) for v in values if isinstance(v, str)}

    def clean_input(self, df):
        """Normalize the name columns to matching keys; capture display maps.

        Only name-bearing fields are normalized: the Leerling index, the Stamgroep column
        and the wish-target 'Waarde' columns. Other fields (Jongen/meisje,
        MinimaleTevredenheid, Gewicht) are left untouched, so case-folding the keys cannot
        corrupt e.g. the sex labels. The working DataFrame ends up keyed by ``matching_key``
        so wishes match students and groups regardless of case or spaces;
        ``student_display`` and ``stamgroep_display`` map those keys back to the name as
        entered, for the report layer.
        """
        self.student_display = self._display_map(df.index)
        df.index = df.index.map(matching_key)

        for col in df.columns:
            type_wens = col[0] if isinstance(col, tuple) else col
            type_waarde = col[2] if isinstance(col, tuple) and len(col) > 2 else None
            if type_wens == "Stamgroep":
                self.stamgroep_display = self._display_map(df[col])
            elif type_waarde != "Waarde":
                continue
            df[col] = df[col].apply(matching_key)
        return df

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
        # This check does not seem to work in pandera (perhaps because
        # of np.nan in the Index)
        expected_columns = pd.MultiIndex.from_tuples(
            VOORKEUREN_SCHEMA.columns.keys(),
            names=["TypeWens", "Nr", "TypeWaarde"],
        )
        validate_columns(df, expected_columns, "preferences")
        return validate_schema_with_filetype(
            df, VOORKEUREN_SCHEMA, filetype="voorkeuren"
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

        try:
            validate_schema_with_filetype(self.df, schema, filetype="voorkeuren")
        except pa.errors.SchemaError as exc:
            # Surface the offending students by the name as entered, not the matching key,
            # so the Dutch error message in the app layer is recognisable to the teacher.
            exc.offending_students = self._students_in_failed_column(exc)
            raise

    def _students_in_failed_column(self, exc) -> list:
        """Names (as entered) of students with a missing value in the failed column.

        Best-effort and side-effect free: used to enrich a SchemaError before it bubbles
        up to the app layer. Returns an empty list when the column is not recognisable.
        """
        column = getattr(exc, "column_name", None)
        if column not in self.df.columns:
            return []
        student_display = getattr(self, "student_display", {})
        missing = self.df.index[self.df[column].isna()].get_level_values("Leerling")
        return list(dict.fromkeys(student_display.get(key, key) for key in missing))

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


def validate_not_together(
    rules: list[dict], students: Iterable, n_groups: int
) -> list[dict]:
    """Validate not-together rules against the known student list and group count.

    Rule groups hold names as entered; matching is done on the ``matching_key`` so case
    and spacing do not matter, while any error message reports the name as the user typed
    it. Works on the list[dict] structure — no xlsx required. Raises ValidationError on
    invalid input; returns rules unchanged when valid.
    """
    known = {matching_key(s) for s in students}
    for i, rule in enumerate(rules, start=1):
        group = rule["group"]
        max_samen = rule["Max_aantal_samen"]
        n_students = len(group)

        if n_students < 2:
            raise ValidationError(
                "too_few_students_not_together",
                context={"rule_index": i},
            )

        if max_samen < 1:
            raise ValidationError(
                "invalid_max_samen_not_together",
                context={"rule_index": i, "max_samen": max_samen},
            )

        unknown = sorted(s for s in group if matching_key(s) not in known)
        if unknown:
            raise ValidationError(
                "unknown_student_not_together",
                context={"unknown_students": ", ".join(unknown)},
            )

        if n_students / max_samen > n_groups:
            raise ValidationError(
                "too_strict_not_together",
                context={
                    "rule_index": i,
                    "n_students": n_students,
                    "max_samen": max_samen,
                    "n_groups": n_groups,
                },
            )
    return rules


def read_groups_excel(path_groups_to) -> tuple[dict, dict]:
    """Read the target groups from excel.

    Returns
    -------
    tuple[dict, dict]
        ``(groups_to, group_display)``. ``groups_to`` is keyed by ``matching_key`` (so it
        matches the keyed wish targets); ``group_display`` maps each key back to the group
        name as entered, for the report layer.
    """
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

    group_display = {matching_key(g): display_name(g) for g in df["Groepen"]}
    groups_to = (
        df.assign(Groepen=lambda df: df["Groepen"].apply(matching_key))
        .set_index("Groepen")
        .to_dict(orient="index")
    )
    return groups_to, group_display


class EdexReader:  # pylint: disable=too-few-public-methods  # data exposed via attributes set in __init__
    """Read EDEX file"""

    def __init__(self, file_loc):
        self.file_loc = file_loc
        self.df_leerlingen = self._parse_leerlingen().pipe(self._clean_leerlingen)
        self.df_groepen = self._parse_groepen().pipe(self._clean_groepen)

    def _parse_leerlingen(self):
        if isinstance(self.file_loc, BytesIO):
            self.file_loc.seek(0)
        tree = ET.parse(self.file_loc)
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
        return pd.DataFrame(rows)

    @staticmethod
    def _clean_leerlingen(df):
        geslacht_code = {
            "0": "Onbekend",
            "1": "Jongen",
            "2": "Meisje",
            "9": "Niet gespecificeerd",
        }

        return (
            df.set_index("key")
            .astype({"jaargroep": "Int64"})
            .assign(geslacht=lambda df: df["geslacht"].map(geslacht_code))
        )

    def _parse_groepen(self):
        if isinstance(self.file_loc, BytesIO):
            self.file_loc.seek(0)

        tree = ET.parse(self.file_loc)
        root = tree.getroot()

        rows = []
        for gg in root.findall("./groepen/groep"):
            data = {}
            data["key"] = gg.attrib.get("key")
            for child in gg:
                data[child.tag] = child.text
            rows.append(data)
        return pd.DataFrame(rows)

    @staticmethod
    def _clean_groepen(df) -> pd.DataFrame:
        return df.set_index("key").astype({"jaargroep": "Int64"})

    def get_full_df(self) -> pd.DataFrame:
        """Enrich students with group information"""
        df = (
            self.df_leerlingen.merge(
                self.df_groepen,
                left_on="groepscode",
                right_index=True,
                suffixes=("", "_groep"),
            )
            .rename(columns={"naam": "groepsnaam"})
            .drop(columns=["jaargroep_groep"])
        )
        return df
