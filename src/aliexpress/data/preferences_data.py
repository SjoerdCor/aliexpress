"""Datacontract bundling everything the solver and reporting need about preferences.

``PreferenceData`` is the canonical in-memory representation that both input paths
(Excel upload and — later — the web form) produce and that the solver consumes. It
serialises losslessly to/from JSON so it can be persisted as ``voorkeuren.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PreferenceData:
    """The artefacts the solver/reporting derive from a set of preferences.

    Attributes
    ----------
    preferences:
        Long-format frame with a ``(Leerling, TypeWens, Nr)`` MultiIndex and the
        columns ``Waarde`` and ``Gewicht``.
    students_info:
        Per matching-key meta info (``MinimaleTevredenheid``, ``Jongen/meisje``,
        ``Stamgroep``), as produced by ``VoorkeurenProcessor.get_students_meta_info``.
    student_display:
        Maps each student matching-key to the name as the user entered it.
    unique_name:
        Maps each student matching-key to a short *unique* display name (roepnaam plus the
        minimal surname letters needed to disambiguate), parallel to ``student_display``.
        Filled only in the web-form path; empty in the Excel path, which has no separate
        roepnaam/achternaam. Consumers fall back to the full name when a key is absent.
    stamgroep_display:
        Maps each stamgroep matching-key to the label as the user entered it.
    input_sheet:
        The original *wide* preferences frame (``VoorkeurenProcessor.input``) with a
        ``(TypeWens, Nr, TypeWaarde)`` MultiIndex on the columns. The reporting layer
        renders the per-student fulfilled-wishes overview from this sheet, so it has to
        travel with the data for a solver run to be reproducible from JSON alone.
    """

    preferences: pd.DataFrame
    students_info: dict
    student_display: dict
    stamgroep_display: dict
    input_sheet: pd.DataFrame
    unique_name: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise to a JSON string, preserving the frame's index names and dtypes."""
        frame = self.preferences.reset_index()
        payload = {
            "preferences": {
                "index_names": list(self.preferences.index.names),
                "column_names": list(self.preferences.columns.names),
                "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
                "records": frame.to_dict("records"),
            },
            "students_info": self.students_info,
            "student_display": self.student_display,
            "unique_name": self.unique_name,
            "stamgroep_display": self.stamgroep_display,
            "input_sheet": _wide_sheet_to_payload(self.input_sheet),
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, data: str) -> "PreferenceData":
        """Reconstruct a ``PreferenceData`` from a string produced by :meth:`to_json`."""
        payload = json.loads(data)
        pref = payload["preferences"]
        # pd.DataFrame([]) produces a frame with no columns, so astype would fail.
        # Provide explicit column names when records is empty.
        frame = (
            pd.DataFrame(pref["records"])
            if pref["records"]
            else pd.DataFrame(columns=list(pref["dtypes"]))
        )
        frame = frame.astype(pref["dtypes"])
        frame = frame.set_index(pref["index_names"])
        # Restore the column-axis name (e.g. "TypeWaarde") that reset_index/to_dict drops,
        # so the round-trip is exact for the long-format frame.
        frame.columns.names = pref["column_names"]
        return cls(
            preferences=frame,
            students_info=payload["students_info"],
            student_display=payload["student_display"],
            unique_name=payload.get("unique_name", {}),
            stamgroep_display=payload["stamgroep_display"],
            input_sheet=_wide_sheet_from_payload(payload["input_sheet"]),
        )


def get_graag_met(preferences: pd.DataFrame) -> pd.DataFrame:
    """Return the 'Graag met' slice of preferences; empty DataFrame when none present.

    Equivalent to preferences.xs("Graag met", level="TypeWens") but safe when
    no positive preferences exist.
    """
    mask = preferences.index.get_level_values("TypeWens") == "Graag met"
    return preferences.loc[mask].droplevel("TypeWens")


def _wide_sheet_to_payload(sheet: pd.DataFrame) -> dict:
    """Serialise the wide input sheet (MultiIndex columns) to a JSON-safe dict.

    ``to_dict(orient="split")`` keeps the index, the column tuples and the row data
    separately, which is exactly what is needed to rebuild the MultiIndex columns. The
    column names and per-column dtypes are stored explicitly so the round-trip is exact;
    NaN sub-levels in the column tuples survive as JSON ``null`` and are restored below.
    """
    split = sheet.to_dict(orient="split")
    return {
        "index": split["index"],
        "index_name": sheet.index.name,
        "columns": [list(col) for col in split["columns"]],
        "column_names": list(sheet.columns.names),
        "data": split["data"],
        "dtypes": [str(dtype) for dtype in sheet.dtypes],
    }


def _wide_sheet_from_payload(payload: dict) -> pd.DataFrame:
    """Reconstruct the wide input sheet from :func:`_wide_sheet_to_payload`."""
    columns = pd.MultiIndex.from_tuples(
        [
            tuple(np.nan if part is None else part for part in col)
            for col in payload["columns"]
        ],
        names=payload["column_names"],
    )
    index = pd.Index(payload["index"], name=payload["index_name"])
    sheet = pd.DataFrame(payload["data"], index=index, columns=columns)
    # Restore per-column dtypes; object columns with NaN keep their original layout.
    sheet = sheet.astype(dict(zip(sheet.columns, payload["dtypes"])))
    return sheet
