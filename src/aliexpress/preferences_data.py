"""Datacontract bundling everything the solver and reporting need about preferences.

``PreferenceData`` is the canonical in-memory representation that both input paths
(Excel upload and — later — the web form) produce and that the solver consumes. It
serialises losslessly to/from JSON so it can be persisted as ``voorkeuren.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd


@dataclass
class PreferenceData:
    """The four artefacts the solver/reporting derive from a set of preferences.

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
    stamgroep_display:
        Maps each stamgroep matching-key to the label as the user entered it.
    """

    preferences: pd.DataFrame
    students_info: dict
    student_display: dict
    stamgroep_display: dict

    def to_json(self) -> str:
        """Serialise to a JSON string, preserving the frame's index names and dtypes."""
        frame = self.preferences.reset_index()
        payload = {
            "preferences": {
                "index_names": list(self.preferences.index.names),
                "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
                "records": frame.to_dict("records"),
            },
            "students_info": self.students_info,
            "student_display": self.student_display,
            "stamgroep_display": self.stamgroep_display,
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, data: str) -> "PreferenceData":
        """Reconstruct a ``PreferenceData`` from a string produced by :meth:`to_json`."""
        payload = json.loads(data)
        pref = payload["preferences"]
        frame = pd.DataFrame(pref["records"])
        frame = frame.astype(pref["dtypes"])
        frame = frame.set_index(pref["index_names"])
        return cls(
            preferences=frame,
            students_info=payload["students_info"],
            student_display=payload["student_display"],
            stamgroep_display=payload["stamgroep_display"],
        )
