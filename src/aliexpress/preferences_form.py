"""Build a :class:`PreferenceData` from web-form wishes (the second input path).

The Excel path bakes arbitrary limits into its wide column schema (5 x "Graag met",
1 x "Liever niet met", 2 x "Niet in"). The web form has no such fixed columns, so this
builder accepts an unbounded number of wishes per type. It is storage-agnostic: it takes
small, readable dataclasses and returns a ``PreferenceData`` object — it does not read or
write files.

Validation is shared with the Excel path: the schema / uniqueness / target-exists checks
go through :func:`datareader.validate_long_preferences` (one source of truth). The extra
bounds that only make sense for the form (weight > 0 with a friendly message,
``minimale_tevredenheid <= 1``, and the "Niet in" cap) are enforced here before / after
that call, raising :class:`errors.ValidationError` with Dutch message keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .datareader import (
    display_name,
    matching_key,
    toggle_negative_weights,
    validate_long_preferences,
)
from .errors import ValidationError
from .preferences_data import PreferenceData


@dataclass
class Wish:
    """A single "graag met" / "liever niet met" wish: a target name and its weight."""

    naam: str  # target name as entered (a student or a group)
    gewicht: float


@dataclass
class StudentWishes:
    """All wishes a single student submitted through the form.

    ``naam`` fields hold the names exactly as entered; the builder normalises them to
    matching keys. ``niet_in`` holds group names only.
    """

    leerling: str
    geslacht: str  # "Jongen" | "Meisje"
    stamgroep: str
    minimale_tevredenheid: float | None
    graag_met: list[Wish] = field(default_factory=list)
    liever_niet_met: list[Wish] = field(default_factory=list)
    niet_in: list[str] = field(default_factory=list)


def build_preference_data(
    students: list[StudentWishes], all_to_groups: list[str]
) -> PreferenceData:
    """Turn submitted form wishes into the canonical ``PreferenceData`` contract.

    Parameters
    ----------
    students : list[StudentWishes]
        One entry per selected student. A student without any wishes is allowed.
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
        On a non-positive weight, ``minimale_tevredenheid > 1``, or a "Niet in" set that
        leaves the student nowhere to go (``>= len(all_to_groups)`` groups).
    pandera.errors.SchemaError
        On a duplicate target within a student or an unknown target (via the shared
        long-format validation).
    """
    _check_niet_in_cap(students, all_to_groups)
    _check_weights_and_min_satisfaction(students)

    student_display, stamgroep_display = _build_display_maps(students)

    long_df = _build_long_frame(students)
    all_leerlingen = [matching_key(s.leerling) for s in students]
    long_df = validate_long_preferences(long_df, all_to_groups, all_leerlingen)

    preferences = toggle_negative_weights(long_df, mask="Liever niet met")
    input_sheet = _build_input_sheet(students)
    students_info = _build_students_info(students)

    return PreferenceData(
        preferences=preferences,
        students_info=students_info,
        student_display=student_display,
        stamgroep_display=stamgroep_display,
        input_sheet=input_sheet,
    )


def _check_niet_in_cap(students: list[StudentWishes], all_to_groups: list[str]) -> None:
    """Reject a "Niet in" set that leaves a student with no group to go to.

    A student may avoid at most ``len(all_to_groups) - 1`` groups; avoiding every group
    makes them unplaceable. Enforced server-side (the form also blocks it client-side).
    """
    cap = len(all_to_groups) - 1
    for student in students:
        if len(student.niet_in) > cap:
            raise ValidationError(
                "too_many_niet_in_form",
                context={
                    "leerling": display_name(student.leerling),
                    "max_niet_in": cap,
                    "n_groepen": len(all_to_groups),
                },
            )


def _check_weights_and_min_satisfaction(students: list[StudentWishes]) -> None:
    """Enforce the form-only bounds: weight > 0 and ``minimale_tevredenheid <= 1``.

    The shared schema also rejects non-positive weights, but checking here first lets us
    name the offending student in a friendly Dutch message. A negative minimal
    satisfaction is deliberately allowed (a student can be forced onto a "liever niet met"
    wish, so the total satisfaction can legitimately be negative).
    """
    for student in students:
        for wish in (*student.graag_met, *student.liever_niet_met):
            if wish.gewicht <= 0:
                raise ValidationError(
                    "invalid_gewicht_form",
                    context={
                        "leerling": display_name(student.leerling),
                        "gewicht": wish.gewicht,
                    },
                )
        min_tev = student.minimale_tevredenheid
        if min_tev is not None and min_tev > 1:
            raise ValidationError(
                "invalid_min_tevredenheid_form",
                context={
                    "leerling": display_name(student.leerling),
                    "minimale_tevredenheid": min_tev,
                },
            )


def _build_display_maps(students: list[StudentWishes]) -> tuple[dict, dict]:
    """Map student and stamgroep matching keys back to the names as entered."""
    student_display = {
        matching_key(s.leerling): display_name(s.leerling) for s in students
    }
    stamgroep_display = {
        matching_key(s.stamgroep): display_name(s.stamgroep) for s in students
    }
    return student_display, stamgroep_display


def _build_long_frame(students: list[StudentWishes]) -> pd.DataFrame:
    """Build the pre-negation long-format frame with matching-key targets.

    Index ``(Leerling, TypeWens, Nr)`` (Nr is a 1-based float per (Leerling, TypeWens),
    matching ``restructure``); columns ``Waarde`` (target key) and ``Gewicht`` (float).
    'Niet in' rows get a placeholder weight of 1.0 (the Excel path has no weight column
    there either; ``restructure`` fills it with 1).
    """
    records = []
    index = []
    for student in students:
        leerling = matching_key(student.leerling)
        for nr, wish in enumerate(student.graag_met, start=1):
            index.append((leerling, "Graag met", float(nr)))
            records.append({"Waarde": matching_key(wish.naam), "Gewicht": wish.gewicht})
        for nr, wish in enumerate(student.liever_niet_met, start=1):
            index.append((leerling, "Liever niet met", float(nr)))
            records.append({"Waarde": matching_key(wish.naam), "Gewicht": wish.gewicht})
        for nr, groep in enumerate(student.niet_in, start=1):
            index.append((leerling, "Niet in", float(nr)))
            records.append({"Waarde": matching_key(groep), "Gewicht": 1.0})

    frame = pd.DataFrame(records, columns=["Waarde", "Gewicht"])
    frame.index = pd.MultiIndex.from_tuples(index, names=["Leerling", "TypeWens", "Nr"])
    frame.columns.names = ["TypeWaarde"]
    return frame


def _build_input_sheet(students: list[StudentWishes]) -> pd.DataFrame:
    """Build the wide input sheet in ``VoorkeurenProcessor.input``'s exact structure.

    Columns are a ``(TypeWens, Nr, TypeWaarde)`` MultiIndex: the three info columns keyed
    by NaN sub-levels, then per wish ``(TypeWens, k, "Waarde")`` (and ``"Gewicht"`` for the
    two weighted types). ``k`` is unbounded. Wishes are the *original* (pre-negation) ones,
    targets stored as matching keys. The reporting layer iterates these columns dynamically.
    """
    max_graag = max((len(s.graag_met) for s in students), default=0)
    max_liever = max((len(s.liever_niet_met) for s in students), default=0)
    max_nietin = max((len(s.niet_in) for s in students), default=0)

    columns = [
        ("MinimaleTevredenheid", np.nan, np.nan),
        ("Jongen/meisje", np.nan, np.nan),
        ("Stamgroep", np.nan, np.nan),
    ]
    for k in range(1, max_graag + 1):
        columns.append(("Graag met", float(k), "Waarde"))
        columns.append(("Graag met", float(k), "Gewicht"))
    for k in range(1, max_liever + 1):
        columns.append(("Liever niet met", float(k), "Waarde"))
        columns.append(("Liever niet met", float(k), "Gewicht"))
    for k in range(1, max_nietin + 1):
        columns.append(("Niet in", float(k), "Waarde"))

    rows = [_input_sheet_row(student, columns) for student in students]
    index = pd.Index([matching_key(s.leerling) for s in students], name="Leerling")
    sheet = pd.DataFrame(rows, index=index, columns=pd.MultiIndex.from_tuples(columns))
    sheet.columns.names = ["TypeWens", "Nr", "TypeWaarde"]
    return sheet


def _input_sheet_row(student: StudentWishes, columns: list) -> list:
    """One wide-sheet row as a list aligned to ``columns``.

    A dict keyed by the column tuples cannot be used because the NaN sub-levels of the
    info columns never compare equal, so pandas would drop those cells. Building the row
    positionally avoids that and keeps the cell order identical to ``columns``.
    """
    cells = {
        ("MinimaleTevredenheid", np.nan, np.nan): (
            np.nan
            if student.minimale_tevredenheid is None
            else float(student.minimale_tevredenheid)
        ),
        ("Jongen/meisje", np.nan, np.nan): student.geslacht,
        ("Stamgroep", np.nan, np.nan): matching_key(student.stamgroep),
    }
    for nr, wish in enumerate(student.graag_met, start=1):
        cells[("Graag met", float(nr), "Waarde")] = matching_key(wish.naam)
        cells[("Graag met", float(nr), "Gewicht")] = float(wish.gewicht)
    for nr, wish in enumerate(student.liever_niet_met, start=1):
        cells[("Liever niet met", float(nr), "Waarde")] = matching_key(wish.naam)
        cells[("Liever niet met", float(nr), "Gewicht")] = float(wish.gewicht)
    for nr, groep in enumerate(student.niet_in, start=1):
        cells[("Niet in", float(nr), "Waarde")] = matching_key(groep)

    return [_cell_lookup(cells, col) for col in columns]


def _cell_lookup(cells: dict, col: tuple):
    """Look up a column's cell, treating NaN-keyed info columns by their first level."""
    if isinstance(col[1], float) and np.isnan(col[1]):
        return next(v for k, v in cells.items() if k[0] == col[0])
    return cells.get(col, np.nan)


def _build_students_info(students: list[StudentWishes]) -> dict:
    """Per matching-key meta info in ``get_students_meta_info``'s shape."""
    return {
        matching_key(student.leerling): {
            "MinimaleTevredenheid": (
                np.nan
                if student.minimale_tevredenheid is None
                else float(student.minimale_tevredenheid)
            ),
            "Jongen/meisje": student.geslacht,
            "Stamgroep": matching_key(student.stamgroep),
        }
        for student in students
    }
