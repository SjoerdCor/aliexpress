"""Build a :class:`PreferenceData` from web-form preferences (the second input path).

The Excel path bakes arbitrary limits into its wide column schema (5 x "Graag met",
1 x "Liever niet met", 2 x "Niet in"). The web form has no such fixed columns, so this
builder accepts an unbounded number of preferences per type. It is storage-agnostic: it
takes small, readable dataclasses and returns a ``PreferenceData`` object — it does not
read or write files.

Validation is shared with the Excel path: the schema / uniqueness / target-exists checks
go through :func:`datareader.validate_long_preferences` (one source of truth). The extra
bounds that only make sense for the form (``min_satisfaction <= 1`` and the "Niet in" cap)
are enforced here, raising :class:`errors.ValidationError` with Dutch message keys. A
non-positive weight is rejected at construction time by :class:`Preference`.
"""

from __future__ import annotations

import numbers
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from ..errors import ValidationError
from .datareader import (
    display_name,
    matching_key,
    toggle_negative_weights,
    validate_long_preferences,
)
from .preferences_data import PreferenceData


class PreferenceKind(Enum):
    """The kind of a preference; the value is the Dutch data string used in the frame."""

    TOGETHER = "Graag met"  # wants to sit with the target
    APART = "Liever niet met"  # would rather not sit with the target


@dataclass
class Preference:
    """A single preference: a target (student or group) with a positive weight and a kind."""

    target: str  # target name as entered (a student or a group)
    weight: float
    kind: PreferenceKind

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError(f"Preference weight must be > 0, got {self.weight}")


@dataclass
class StudentEntry:
    """One student's full form input: identity, meta and preferences.

    ``student``, ``origin_group`` and each preference ``target`` hold the names exactly as
    entered; the builder normalises them to matching keys. ``excluded_groups`` holds group
    names only. ``year_group`` is ``None`` in doorzetten mode, where students have no
    year cohort.
    """

    student: str
    sex: str  # "Jongen" | "Meisje" (data values stay Dutch)
    origin_group: str  # the student's current group
    min_satisfaction: float | None
    year_group: int | None = None
    _: KW_ONLY
    preferences: list[Preference] = field(default_factory=list)
    excluded_groups: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Integral (not int) so pandas/numpy integers from the wizard pass too.
        if self.year_group is not None and not isinstance(
            self.year_group, numbers.Integral
        ):
            raise TypeError(
                f"year_group must be a whole number or None, got {self.year_group!r}"
            )


def build_preference_data(
    students: list[StudentEntry], all_to_groups: list[str]
) -> PreferenceData:
    """Turn submitted form preferences into the canonical ``PreferenceData`` contract.

    Parameters
    ----------
    students : list[StudentEntry]
        One entry per selected student. A student without any preferences is allowed.
    all_to_groups : list[str]
        Matching keys of the destination groups (the route passes
        ``list(target_groups.counts.keys())``).

    Returns
    -------
    PreferenceData
        ``preferences`` is the long-format frame *post-negation* ("Liever niet met" has a
        negative weight); ``input_sheet`` is the wide, pre-negation frame mirroring
        ``VoorkeurenProcessor.input``.

    Raises
    ------
    errors.ValidationError
        On ``min_satisfaction > 1`` or an ``excluded_groups`` set that leaves the student
        nowhere to go (``>= len(all_to_groups)`` groups).
    pandera.errors.SchemaError
        On a duplicate target within a student or an unknown target (via the shared
        long-format validation).
    """
    _check_excluded_groups_cap(students, all_to_groups)
    _check_min_satisfaction(students)

    student_display, origin_group_display = _build_display_maps(students)

    long_df = _build_long_frame(students)
    all_students = [matching_key(s.student) for s in students]
    long_df = validate_long_preferences(long_df, all_to_groups, all_students)

    preferences = toggle_negative_weights(long_df, mask="Liever niet met")
    input_sheet = _build_input_sheet(students)
    students_info = _build_students_info(students)

    return PreferenceData(
        preferences=preferences,
        students_info=students_info,
        student_display=student_display,
        stamgroep_display=origin_group_display,
        input_sheet=input_sheet,
    )


def _check_excluded_groups_cap(
    students: list[StudentEntry], all_to_groups: list[str]
) -> None:
    """Reject an ``excluded_groups`` set that leaves a student with no group to go to.

    A student may avoid at most ``len(all_to_groups) - 1`` groups; avoiding every group
    makes them unplaceable. This needs cross-object context (all destination groups), so it
    cannot live on the dataclass. Enforced server-side (the form also blocks it client-side).
    """
    cap = len(all_to_groups) - 1
    for student in students:
        if len(student.excluded_groups) > cap:
            raise ValidationError(
                "too_many_niet_in_form",
                context={
                    "leerling": display_name(student.student),
                    "max_niet_in": cap,
                    "n_groepen": len(all_to_groups),
                },
            )


def _check_min_satisfaction(students: list[StudentEntry]) -> None:
    """Enforce the form-only bound ``min_satisfaction <= 1`` with a friendly Dutch message.

    A negative minimal satisfaction is deliberately allowed (a student can be forced onto a
    "liever niet met" preference, so the total satisfaction can legitimately be negative).
    """
    for student in students:
        min_sat = student.min_satisfaction
        if min_sat is not None and min_sat > 1:
            raise ValidationError(
                "invalid_min_tevredenheid_form",
                context={
                    "leerling": display_name(student.student),
                    "minimale_tevredenheid": min_sat,
                },
            )


def _build_display_maps(students: list[StudentEntry]) -> tuple[dict, dict]:
    """Map student and origin-group matching keys back to the names as entered."""
    student_display = {
        matching_key(s.student): display_name(s.student) for s in students
    }
    origin_group_display = {
        matching_key(s.origin_group): display_name(s.origin_group) for s in students
    }
    return student_display, origin_group_display


def _build_long_frame(students: list[StudentEntry]) -> pd.DataFrame:
    """Build the pre-negation long-format frame with matching-key targets.

    Index ``(Leerling, TypeWens, Nr)`` (Nr is a 1-based float per (Leerling, TypeWens),
    matching ``restructure``); columns ``Waarde`` (target key) and ``Gewicht`` (float).
    Preferences are grouped by kind and renumbered per kind. 'Niet in' rows get a
    placeholder weight of 1.0 (the Excel path has no weight column there either;
    ``restructure`` fills it with 1).
    """
    records = []
    index = []
    for student in students:
        student_key = matching_key(student.student)
        counters: dict[PreferenceKind, int] = {}
        for preference in student.preferences:
            nr = counters.get(preference.kind, 0) + 1
            counters[preference.kind] = nr
            index.append((student_key, preference.kind.value, float(nr)))
            records.append(
                {
                    "Waarde": matching_key(preference.target),
                    "Gewicht": preference.weight,
                }
            )
        for nr, group in enumerate(student.excluded_groups, start=1):
            index.append((student_key, "Niet in", float(nr)))
            records.append({"Waarde": matching_key(group), "Gewicht": 1.0})

    frame = pd.DataFrame(records, columns=["Waarde", "Gewicht"])
    frame.index = pd.MultiIndex.from_tuples(index, names=["Leerling", "TypeWens", "Nr"])
    frame.columns.names = ["TypeWaarde"]
    return frame


def _build_input_sheet(students: list[StudentEntry]) -> pd.DataFrame:
    """Build the wide input sheet in ``VoorkeurenProcessor.input``'s exact structure.

    Columns are a ``(TypeWens, Nr, TypeWaarde)`` MultiIndex: the three info columns keyed
    by NaN sub-levels, then per preference ``(TypeWens, k, "Waarde")`` (and ``"Gewicht"`` for
    the two weighted kinds). ``k`` is unbounded. Preferences are the *original*
    (pre-negation) ones, targets stored as matching keys. The reporting layer iterates these
    columns dynamically.
    """
    max_together = max(
        (_count_kind(s, PreferenceKind.TOGETHER) for s in students), default=0
    )
    max_apart = max((_count_kind(s, PreferenceKind.APART) for s in students), default=0)
    max_excluded = max((len(s.excluded_groups) for s in students), default=0)

    columns = [
        ("MinimaleTevredenheid", np.nan, np.nan),
        ("Jongen/meisje", np.nan, np.nan),
        ("Stamgroep", np.nan, np.nan),
    ]
    for k in range(1, max_together + 1):
        columns.append(("Graag met", float(k), "Waarde"))
        columns.append(("Graag met", float(k), "Gewicht"))
    for k in range(1, max_apart + 1):
        columns.append(("Liever niet met", float(k), "Waarde"))
        columns.append(("Liever niet met", float(k), "Gewicht"))
    for k in range(1, max_excluded + 1):
        columns.append(("Niet in", float(k), "Waarde"))

    rows = [_input_sheet_row(student, columns) for student in students]
    index = pd.Index([matching_key(s.student) for s in students], name="Leerling")
    sheet = pd.DataFrame(rows, index=index, columns=pd.MultiIndex.from_tuples(columns))
    sheet.columns.names = ["TypeWens", "Nr", "TypeWaarde"]
    return sheet


def _count_kind(student: StudentEntry, kind: PreferenceKind) -> int:
    """Number of preferences of a given kind for one student."""
    return sum(1 for p in student.preferences if p.kind is kind)


def _input_sheet_row(student: StudentEntry, columns: list) -> list:
    """One wide-sheet row as a list aligned to ``columns``.

    A dict keyed by the column tuples cannot be used because the NaN sub-levels of the
    info columns never compare equal, so pandas would drop those cells. Building the row
    positionally avoids that and keeps the cell order identical to ``columns``.
    """
    cells = {
        ("MinimaleTevredenheid", np.nan, np.nan): (
            np.nan
            if student.min_satisfaction is None
            else float(student.min_satisfaction)
        ),
        ("Jongen/meisje", np.nan, np.nan): student.sex,
        ("Stamgroep", np.nan, np.nan): matching_key(student.origin_group),
    }
    counters: dict[PreferenceKind, int] = {}
    for preference in student.preferences:
        nr = counters.get(preference.kind, 0) + 1
        counters[preference.kind] = nr
        type_wens = preference.kind.value
        cells[(type_wens, float(nr), "Waarde")] = matching_key(preference.target)
        cells[(type_wens, float(nr), "Gewicht")] = float(preference.weight)
    for nr, group in enumerate(student.excluded_groups, start=1):
        cells[("Niet in", float(nr), "Waarde")] = matching_key(group)

    return [_cell_lookup(cells, col) for col in columns]


def _cell_lookup(cells: dict, col: tuple):
    """Look up a column's cell, treating NaN-keyed info columns by their first level."""
    if isinstance(col[1], float) and np.isnan(col[1]):
        return next(v for k, v in cells.items() if k[0] == col[0])
    return cells.get(col, np.nan)


def _build_students_info(students: list[StudentEntry]) -> dict:
    """Per matching-key meta info in ``get_students_meta_info``'s shape."""
    result = {}
    for student in students:
        info = {
            "MinimaleTevredenheid": (
                np.nan
                if student.min_satisfaction is None
                else float(student.min_satisfaction)
            ),
            "Jongen/meisje": student.sex,
            "Stamgroep": matching_key(student.origin_group),
        }
        if student.year_group is not None:
            info["Jaarlaag"] = student.year_group
        result[matching_key(student.student)] = info
    return result
