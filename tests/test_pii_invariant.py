"""Regression test: student names must never appear in the aliexpress package logger."""

# The local test doubles are deliberately minimal and access the worker's private
# error boundary to assert the technical-log invariant.
# pylint: disable=duplicate-code,missing-class-docstring,missing-function-docstring,protected-access,too-few-public-methods

import shutil
from pathlib import Path

from aliexpress import errors
from aliexpress.data import datareader
from aliexpress.web import tasks
from aliexpress.web.extensions import db as flask_db
from aliexpress.web.models import Process
from aliexpress.web.process_files import save_voorkeuren
from app import app as flask_app

_INTEGRATION = Path(__file__).parent / "integration"

_KNOWN_NAMES = ["Anna", "Bram", "Claire", "Daan", "Eva"]


def test_no_student_names_logged_on_not_together_get(
    client, tmp_path, captured_aliexpress_logs
):
    """GET /not_together must not log any student name at any level."""
    proc_dir = tmp_path / "test-school" / "pii-test"
    proc_dir.mkdir(parents=True)
    shutil.copy(_INTEGRATION / "groepen_small.xlsx", proc_dir / "groups.xlsx")

    groups_to, _ = datareader.read_groups_excel(str(proc_dir / "groups.xlsx"))
    processor = datareader.VoorkeurenProcessor(_INTEGRATION / "voorkeuren_small.xlsx")
    processor.process(all_to_groups=list(groups_to.keys()))
    with flask_app.app_context():
        save_voorkeuren(
            "test-school", "pii-test", processor.to_preference_data(), source="excel"
        )
        flask_db.session.add(Process(school_id="test-school", name="pii-test"))
        flask_db.session.commit()

    client.get("/processes/select/pii-test")
    client.get("/not_together")

    combined = " ".join(captured_aliexpress_logs)
    for name in _KNOWN_NAMES:
        assert name not in combined, f"Student name {name!r} leaked into the log"


def test_detailed_conflict_names_are_not_written_to_technical_log(
    captured_aliexpress_logs, monkeypatch
):
    """The display context can contain names without putting them in the exception log."""

    class DummyRun:
        def set_status(self, *_args):
            pass

    class DummyProcess:
        run = DummyRun()

    monkeypatch.setattr(tasks.Process, "by_name", lambda *_args: DummyProcess())
    exc = errors.FeasibilityError(
        "infeasible_preferences",
        context={
            "case": "detailed",
            "conflict": {
                "conditions": [
                    {
                        "type": "minimum_satisfaction",
                        "student": "Anna Jansen",
                        "floor": 1.0,
                        "preferences": [
                            {
                                "kind": "Graag met",
                                "target": "Bram Visser",
                                "weight": 1.0,
                            }
                        ],
                    }
                ]
            },
        },
        technical_message="Hard preference constraints are mutually infeasible",
    )

    try:
        raise exc
    except errors.FeasibilityError as caught:
        tasks._handle_failure(caught, "school", "proces")

    combined = " ".join(captured_aliexpress_logs)
    assert "Anna Jansen" not in combined
    assert "Bram Visser" not in combined
