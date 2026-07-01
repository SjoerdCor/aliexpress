"""Regression test: student names must never appear in the aliexpress package logger."""

import shutil
from pathlib import Path

from aliexpress.data import datareader
from aliexpress.web.extensions import db as flask_db
from aliexpress.web.models import Process
from aliexpress.web.routes.wizard import _write_voorkeuren_json
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
    _write_voorkeuren_json(
        str(proc_dir / "voorkeuren.json"),
        processor.to_preference_data(),
        source="excel",
    )

    with flask_app.app_context():
        flask_db.session.add(Process(school_id="test-school", name="pii-test"))
        flask_db.session.commit()

    client.get("/processes/select/pii-test")
    client.get("/not_together")

    combined = " ".join(captured_aliexpress_logs)
    for name in _KNOWN_NAMES:
        assert name not in combined, f"Student name {name!r} leaked into the log"
