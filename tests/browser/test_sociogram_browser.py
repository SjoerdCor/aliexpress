"""Browser tracer for the first Cytoscape sociogram slice."""

import json

import pytest

from aliexpress.data import datareader
from aliexpress.web.extensions import db as flask_db
from aliexpress.web.models import Process
from aliexpress.web.process_files import save_voorkeuren
from app import app
from tests.browser.conftest import TEST_SCHOOLCODE


@pytest.mark.usefixtures("login")
def test_sociogram_renders_real_nodes_and_directed_preferences(
    live_server, tmp_path, page
):
    """A real process produces Cytoscape nodes and preference endpoints in the browser."""
    proc = tmp_path / TEST_SCHOOLCODE / "sociogramrun"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "relevant_students_and_groups.json").write_text(
        json.dumps({"candidates": [], "groups_from": []}), encoding="utf-8"
    )
    processor = datareader.VoorkeurenProcessor("testdata/voorkeuren_klein.xlsx")
    processor.process(["blauw", "groen", "geel", "oranje"])
    with app.app_context():
        save_voorkeuren(
            TEST_SCHOOLCODE,
            "sociogramrun",
            processor.to_preference_data(),
            source="excel",
        )
        flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name="sociogramrun"))
        flask_db.session.commit()

    page.goto(f"{live_server}/processes/select/sociogramrun")
    page.goto(f"{live_server}/sociogram")
    page.wait_for_function("() => window.sociogramSnapshot !== undefined")
    snapshot = page.evaluate("window.sociogramSnapshot()")

    assert len(snapshot["nodes"]) == 4
    assert len(snapshot["preferences"]) == 1
    assert page.locator("#sociogram canvas").count() >= 1
