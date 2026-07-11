"""Shared test helpers — not fixtures; imported directly by per-blueprint test modules."""

# pylint: disable=duplicate-code  # setup_process is intentionally identical to the copy in
# test_results.py; that file is frozen and cannot import from here.

import json
from unittest.mock import MagicMock

from aliexpress.data.preferences_form import (
    Preference,
    PreferenceKind,
    StudentEntry,
    build_preference_data,
)
from aliexpress.solver.groepsindeling_view import (
    BalanceRow,
    GroepsindelingView,
    GroupCard,
    SexColumn,
    StudentChip,
    YearSection,
)
from aliexpress.web.extensions import db
from aliexpress.web.models import Process
from app import app as flask_app

# Must match the schoolcode created in tests/conftest.py's ``client`` fixture.
SCHOOL_ID = "test-school"


def immediate_thread(target, args=()):
    """Thread replacement whose ``start()`` runs the target synchronously, so route-spawned
    background work finishes before the request returns and is deterministic to assert on.
    """
    runner = MagicMock()
    runner.start.side_effect = lambda: target(*args)
    return runner


def flashes(client_obj):
    """Return list of (category, message) flash tuples from the current session."""
    with client_obj.session_transaction() as sess:
        return sess.get("_flashes", [])


def setup_process(client, tmp_path, process_id="testproces"):
    """Create a process directory, a Process DB row, and set session process_id.

    The directory is placed under the school's subdirectory so it matches
    ``get_process_path(school_id, process_id)`` in the real routes.
    """
    proc_dir = tmp_path / SCHOOL_ID / process_id
    proc_dir.mkdir(parents=True, exist_ok=True)
    with flask_app.app_context():
        proc = Process(school_id=SCHOOL_ID, name=process_id)
        db.session.add(proc)
        db.session.commit()
    with client.session_transaction() as sess:
        sess["process_id"] = process_id
    return proc_dir


def make_process_row(school_id, name):
    """Create a Process DB row and return it (must be called inside an app context)."""
    proc = Process(school_id=school_id, name=name)
    db.session.add(proc)
    db.session.commit()
    return proc


TWO_STUDENTS_GROEN = [
    {
        "key": "s1",
        "roepnaam": "Anna",
        "achternaam": "Bos",
        "groepsnaam": "Groen",
        "geslacht": "Meisje",
    },
    {
        "key": "s2",
        "roepnaam": "Bram",
        "achternaam": "Dijk",
        "groepsnaam": "Groen",
        "geslacht": "Jongen",
    },
]


def make_interim_view():
    """A minimal but real GroepsindelingView: one group, one jaarlaag, one chip.

    Shared by the /interim_result route test and the processing-page browser test so the
    fixture doesn't drift (and pylint's duplicate-code check stays quiet).
    """
    chip = StudentChip(
        chip_name="Anna",
        full_name="Anna Jansen",
        origin_abbrev="Kla",
        origin_full="Klas A",
        year_group=6,
        satisfaction=1.0,
        preferences=[],
        not_in=[],
        min_satisfaction=None,
    )
    boys = SexColumn(sex="jongen", new_count=0, students=[])
    girls = SexColumn(sex="meisje", new_count=1, students=[chip])
    section = YearSection(year=6, label="Jaarlaag 6", size=1, boys=boys, girls=girls)
    card = GroupCard(
        name="Groep 1", total=1, boys_total=0, girls_total=1, year_sections=[section]
    )
    balance_row = BalanceRow(
        label="Totaal",
        is_total=True,
        per_group={"Groep 1": (1, 0, 1)},
        size_diff=0,
        sex_imbalance=1,
    )
    return GroepsindelingView(
        group_order=["Groep 1"], groups=[card], balance_rows=[balance_row]
    )


def write_groups_to_json(proc_dir, groups_to):
    """Persist a candidates JSON whose groups_to maps each group to student dicts."""
    (proc_dir / "relevant_students_and_groups.json").write_text(
        json.dumps({"groups_to": groups_to}), encoding="utf-8"
    )


def make_students(*genders):
    """Build a list of minimal student dicts with the given genders, in order."""
    return [{"geslacht": sex, "roepnaam": "x", "achternaam": "y"} for sex in genders]


def write_minimal_voorkeuren_json(
    proc_dir,
    students=None,
    all_to_groups=None,
    source="form",
):
    """Write a minimal but valid voorkeuren.json to proc_dir for use in route tests.

    Defaults to two students (Alice/Jongen and Bob/Meisje) and two groups (klas a, klas b).
    Pass ``students`` as a list of ``StudentEntry`` objects to override.
    """
    if all_to_groups is None:
        all_to_groups = ["klas a", "klas b"]
    if students is None:
        students = [
            StudentEntry(
                student="Alice",
                sex="Jongen",
                origin_group="Groep 4",
                min_satisfaction=None,
                preferences=[
                    Preference(target="Bob", weight=1.0, kind=PreferenceKind.TOGETHER)
                ],
            ),
            StudentEntry(
                student="Bob",
                sex="Meisje",
                origin_group="Groep 4",
                min_satisfaction=None,
            ),
        ]
    preference_data = build_preference_data(students, all_to_groups)
    payload = json.loads(preference_data.to_json())
    payload["source"] = source
    (proc_dir / "voorkeuren.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
