# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

"""Browser test for the single forward path on the preferences page."""

import json

import pytest


def _open_preferences(live_server, tmp_path, page):
    """Land the browser on the preferences page (groups done, nothing uploaded yet)."""
    proc = tmp_path / "browsertest"
    proc.mkdir(exist_ok=True)
    (proc / "relevant_students_and_groups.json").write_text(
        json.dumps({"candidates": [], "groups_from": []}), encoding="utf-8"
    )
    # select_process lands on student_preferences once groups.xlsx exists.
    (proc / "groups.xlsx").write_bytes(b"dummy")
    page.goto(f"{live_server}/processes/select/browsertest")
    page.wait_for_url("**/student_preferences")
    return proc


@pytest.mark.usefixtures("login")
def test_forward_button_submits_upload_form_and_requires_a_file(
    live_server, tmp_path, page
):
    """The bottom 'Naar Niet samen →' submits the upload form; without a file and without
    an earlier upload it must not advance, but flash a friendly message instead."""
    _open_preferences(live_server, tmp_path, page)

    page.click(".step-navigation button.next-step")
    page.wait_for_url("**/student_preferences")  # stayed on the page

    assert page.locator(".flash-message").inner_text().strip() != ""
    assert "Upload eerst" in page.locator(".flash-message").inner_text()
