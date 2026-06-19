"""Shared test helpers — not fixtures; imported directly by per-blueprint test modules."""

# pylint: disable=duplicate-code  # setup_process is intentionally identical to the copy in
# test_results.py; that file is frozen and cannot import from here.

import json
from unittest.mock import MagicMock

from aliexpress.extensions import db
from aliexpress.models import Process
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


def write_groups_to_json(proc_dir, groups_to):
    """Persist a candidates JSON whose groups_to maps each group to student dicts."""
    (proc_dir / "relevant_students_and_groups.json").write_text(
        json.dumps({"groups_to": groups_to}), encoding="utf-8"
    )


def make_students(*genders):
    """Build a list of minimal student dicts with the given genders, in order."""
    return [{"geslacht": sex, "roepnaam": "x", "achternaam": "y"} for sex in genders]
