"""Typed load/save helpers for per-process wizard artifacts.

Each public function takes ``(school_id, process_id)`` and resolves the file path itself
via :func:`storage.get_file_path`, so route modules never name a file on disk. Uniform
verbs: ``load_*`` / ``save_*`` / ``has_*``. This module may import from ``data`` (the
canonical ``PreferenceData`` format and ``datareader``); ``storage`` stays pure paths.
"""

import json
import os
from io import BytesIO

import pandas as pd

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


def load_not_together(school_id, process_id) -> list[dict]:
    """Load not-together rules from not_together.json, with ``group`` already a set.

    Returns an empty list when no rules were saved yet.
    """
    path = get_file_path(school_id, process_id, "not_together.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [
        {"group": set(r["group"]), "Max_aantal_samen": r["Max_aantal_samen"]}
        for r in raw
    ]


def save_not_together(school_id, process_id, rules) -> None:
    """Persist not-together rules as JSON (sets serialised as lists)."""
    data = [
        {"group": list(r["group"]), "Max_aantal_samen": r["Max_aantal_samen"]}
        for r in rules
    ]
    path = get_file_path(school_id, process_id, "not_together.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


def load_input_method(school_id, process_id) -> str:
    """Load the chosen preferences-input method ("form" or "excel"), defaulting to "form"."""
    path = get_file_path(school_id, process_id, "input_method.json")
    if not os.path.exists(path):
        return "form"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("method", "form")


def save_input_method(school_id, process_id, method: str) -> None:
    """Persist the chosen preferences-input method as input_method.json."""
    path = get_file_path(school_id, process_id, "input_method.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"method": method}, f)


def load_groups(school_id, process_id):
    """Load the destination groups from groups.xlsx: (groups_to, group_display)."""
    path = get_file_path(school_id, process_id, "groups.xlsx")
    return datareader.read_groups_excel(path)


def save_groups_excel(school_id, process_id, distribution) -> None:
    """Persist a group name → occupancy mapping as groups.xlsx."""
    path = get_file_path(school_id, process_id, "groups.xlsx")
    pd.DataFrame(distribution).transpose().to_excel(path, index_label="Groepen")


def load_groups_to(school_id, process_id) -> dict:
    """Load the groups-to mapping (groupname → students) from the candidates JSON."""
    path = get_file_path(school_id, process_id, "relevant_students_and_groups.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("groups_to", {})


def load_groups_to_state(school_id, process_id):
    """Load the saved groups-to form state, or None when the page was not filled yet."""
    path = get_file_path(school_id, process_id, "groups_to_state.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_groups_to_state(school_id, process_id, state) -> None:
    """Persist the groups-to form state (disabled/new groups) as groups_to_state.json."""
    path = get_file_path(school_id, process_id, "groups_to_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def save_candidates(school_id, process_id, data) -> None:
    """Persist the candidates JSON (relevant_students_and_groups.json)."""
    path = get_file_path(school_id, process_id, "relevant_students_and_groups.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_candidates(school_id, process_id):
    """Load (candidate dicts, groups_from, jaargroepen) from relevant_students_and_groups.json.

    ``jaargroepen`` (herindelen only) is the set of jaargroepen settled by the group
    selection itself, not re-derived from the candidates — see ``_select_groups_post``.
    """
    path = get_file_path(school_id, process_id, "relevant_students_and_groups.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return (
        raw.get("candidates", []),
        raw.get("groups_from", []),
        raw.get("jaargroepen", []),
    )


def load_roster(school_id, process_id):
    """Load the saved roster dict, or None when the step was not used yet."""
    path = get_file_path(school_id, process_id, "roster.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_roster(school_id, process_id, data) -> None:
    """Persist the roster (participants) as roster.json."""
    path = get_file_path(school_id, process_id, "roster.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


def save_edexml(school_id, process_id, edex_file) -> None:
    """Persist the uploaded EDEXML file storage object as edex.xml."""
    path = get_file_path(school_id, process_id, "edex.xml")
    edex_file.save(path)


def load_edexml(school_id, process_id) -> BytesIO:
    """Load the previously uploaded EDEXML file as an in-memory BytesIO."""
    path = get_file_path(school_id, process_id, "edex.xml")
    with open(path, "rb") as fh:
        return BytesIO(fh.read())


def has_edexml(school_id, process_id) -> bool:
    """Return whether an edex.xml was uploaded earlier for this process."""
    return os.path.exists(get_file_path(school_id, process_id, "edex.xml"))


_DOWNSTREAM_WIZARD_FILES = (
    "relevant_students_and_groups.json",
    "roster.json",
    "groups.xlsx",
    "groups_to_state.json",
    "input_method.json",
    "preferences_form_state.json",
    "voorkeuren.json",
    "preferences.xlsx",
    "not_together.json",
)


def reset_downstream_wizard_files(school_id, process_id) -> None:
    """Remove wizard artifacts derived from a previous EDEXML upload.

    A fresh upload defines a new population, so any roster / groups / preferences saved
    against the previous upload are stale — the newest upload overwrites them, and the
    wizard restarts cleanly from this point (all three modes).
    """
    for name in _DOWNSTREAM_WIZARD_FILES:
        path = get_file_path(school_id, process_id, name)
        if os.path.exists(path):
            os.remove(path)


def reset_result_files(school_id, process_id) -> None:
    """Remove stale solver output files before starting a new distribution run."""
    for name in ("results.xlsx", "result_tables.json", "sociogram.html"):
        path = get_file_path(school_id, process_id, name)
        if os.path.exists(path):
            os.remove(path)
