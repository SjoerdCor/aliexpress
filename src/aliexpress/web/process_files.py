"""Typed load/save helpers for per-process wizard artifacts.

Each public function takes ``(school_id, process_id)`` and resolves the file path itself
via :func:`storage.get_file_path`, so route modules never name a file on disk. Uniform
verbs: ``load_*`` / ``save_*`` / ``has_*``. This module may import from ``data`` (the
canonical ``PreferenceData`` format and ``datareader``); ``storage`` stays pure paths.
"""

import json
import os

from ..data import datareader
from ..data.preferences_data import PreferenceData
from .storage import get_file_path


def save_voorkeuren(
    school_id, process_id, preference_data: PreferenceData, source: str
) -> None:
    """Persist a PreferenceData as voorkeuren.json, tagged with its input source."""
    path = get_file_path(school_id, process_id, "voorkeuren.json")
    payload = json.loads(preference_data.to_json())
    payload["source"] = source
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def load_voorkeuren(school_id, process_id) -> tuple[PreferenceData, str]:
    """Load a PreferenceData and its source tag from voorkeuren.json."""
    path = get_file_path(school_id, process_id, "voorkeuren.json")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    source = payload.pop("source", "form")
    return PreferenceData.from_json(json.dumps(payload)), source


def load_pref_form_state(school_id, process_id):
    """Load saved form state dict, or None when none exists."""
    path = get_file_path(school_id, process_id, "preferences_form_state.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_pref_form_state(school_id, process_id, state) -> None:
    """Persist the intermediate preferences draft (``preferences_form_state.json``)."""
    path = get_file_path(school_id, process_id, "preferences_form_state.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)


def load_student_names(school_id, process_id, groups_to) -> list[str]:
    """Return sorted display names of students to populate the not-together dropdown.

    Prefers ``voorkeuren.json`` (canonical, written by both input paths); falls back to
    reading the raw Excel for processes created before ``voorkeuren.json`` was introduced.
    """
    if os.path.exists(get_file_path(school_id, process_id, "voorkeuren.json")):
        preference_data, _ = load_voorkeuren(school_id, process_id)
        names = sorted(preference_data.student_display.values())
    else:
        preferences_path = get_file_path(school_id, process_id, "preferences.xlsx")
        processor = datareader.VoorkeurenProcessor(preferences_path)
        processor.process(all_to_groups=list(groups_to.keys()))
        names = sorted(processor.student_display.values())
    return names


def save_preferences_excel(school_id, process_id, raw: bytes) -> None:
    """Persist the raw uploaded preferences workbook as preferences.xlsx.

    Storing the original upload (rather than a re-serialisation) preserves names as
    entered, so re-reading later maps student_display to the correct display names.
    """
    path = get_file_path(school_id, process_id, "preferences.xlsx")
    with open(path, "wb") as fh:
        fh.write(raw)


def has_preferences_excel(school_id, process_id) -> bool:
    """Return whether a preferences.xlsx was uploaded earlier for this process."""
    return os.path.exists(get_file_path(school_id, process_id, "preferences.xlsx"))
