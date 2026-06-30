# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

"""Browser test for the single forward path on the preferences page."""

import json

import pytest

from aliexpress.web.extensions import db as flask_db
from aliexpress.web.models import Process
from app import app
from tests.browser.conftest import TEST_SCHOOLCODE


def _open_preferences(live_server, tmp_path, page):
    """Land the browser on the preferences page (groups done, nothing uploaded yet)."""
    proc = tmp_path / TEST_SCHOOLCODE / "browsertest"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "relevant_students_and_groups.json").write_text(
        json.dumps({"candidates": [], "groups_from": []}), encoding="utf-8"
    )
    # With the roster + groups steps done and the Excel method chosen, select_process
    # resumes on preferences_excel (groups.xlsx present → input_method decides; ADR 0006).
    (proc / "groups.xlsx").write_bytes(b"dummy")
    (proc / "roster.json").write_text(
        json.dumps({"participants": []}), encoding="utf-8"
    )
    (proc / "input_method.json").write_text(
        json.dumps({"method": "excel"}), encoding="utf-8"
    )
    with app.app_context():
        flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name="browsertest"))
        flask_db.session.commit()
    page.goto(f"{live_server}/processes/select/browsertest")
    page.wait_for_url("**/preferences_excel")
    return proc


@pytest.mark.usefixtures("login")
def test_forward_button_submits_upload_form_and_requires_a_file(
    live_server, tmp_path, page
):
    """The bottom 'Naar Niet samen →' submits the upload form; without a file and without
    an earlier upload it must not advance, but flash a friendly message instead."""
    _open_preferences(live_server, tmp_path, page)

    page.click(".step-navigation button.next-step")
    page.wait_for_url("**/preferences_excel")  # stayed on the page

    assert page.locator(".flash-message").inner_text().strip() != ""
    assert "Upload eerst" in page.locator(".flash-message").inner_text()
