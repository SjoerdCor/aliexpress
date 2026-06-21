# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

"""Browser tests for the roster page ("Wie gaat mee"): the JavaScript that the Flask test
client cannot exercise — confirming a new student into a chip, the edit/remove actions, the
collision check, and the unconfirmed-row submit guard.
"""

import json

import pandas as pd
import pytest

from aliexpress.extensions import db as flask_db
from aliexpress.models import Process
from app import app
from tests.browser.conftest import TEST_SCHOOLCODE

CANDIDATES = [
    {
        "key": "s1",
        "roepnaam": "Anna",
        "achternaam": "Bos",
        "geslacht": "Meisje",
        "groepsnaam": "Groep 3",
    },
    {
        "key": "s2",
        "roepnaam": "Bram",
        "achternaam": "Dijk",
        "geslacht": "Jongen",
        "groepsnaam": "Groep 3",
    },
]


def _open_roster(live_server, tmp_path, page):
    """Set up a process (groups done, no roster yet) and land the browser on /roster."""
    proc = tmp_path / TEST_SCHOOLCODE / "browsertest"
    proc.mkdir(parents=True, exist_ok=True)
    # groups_from ends with "Anders" exactly as candidatedetermination produces it.
    (proc / "relevant_students_and_groups.json").write_text(
        json.dumps({"candidates": CANDIDATES, "groups_from": ["Groep 3", "Anders"]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"Jongens": [1, 1], "Meisjes": [1, 0]},
        index=pd.Index(["Klas A", "Klas B"], name="Groepen"),
    ).to_excel(proc / "groups.xlsx")
    with app.app_context():
        flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name="browsertest"))
        flask_db.session.commit()
    page.goto(f"{live_server}/processes/select/browsertest")
    page.wait_for_url("**/roster")
    return proc


def _add_student(page, voornaam, achternaam, geslacht="Meisje", confirm=True):
    """Add one new-student row, fill it, and optionally click 'Toevoegen'."""
    page.click("button:has-text('Leerling toevoegen')")
    row = page.locator(".new-student-row").last
    row.locator("[name='new_voornaam[]']").fill(voornaam)
    row.locator("[name='new_achternaam[]']").fill(achternaam)
    if geslacht:
        row.locator("[name='new_geslacht[]']").select_option(geslacht)
    if confirm:
        row.locator("button.ns-confirm").click()
    return row


@pytest.mark.usefixtures("login")
def test_new_student_becomes_chip_after_confirm(live_server, tmp_path, page):
    """Filling a row and clicking 'Toevoegen' collapses it into a compact chip."""
    _open_roster(live_server, tmp_path, page)
    row = _add_student(page, "Emma", "Jansen")
    assert row.get_attribute("data-confirmed") == "1"
    assert row.locator(".ns-edit").is_hidden()
    chip = row.locator(".ns-chip")
    assert chip.is_visible()
    assert "Emma Jansen" in chip.inner_text()
    assert "Meisje" in chip.inner_text()


@pytest.mark.usefixtures("login")
def test_origin_group_dropdown_has_single_anders(live_server, tmp_path, page):
    """groups_from already ends with 'Anders', so the dropdown must show it only once."""
    _open_roster(live_server, tmp_path, page)
    page.click("button:has-text('Leerling toevoegen')")
    row = page.locator(".new-student-row").last
    anders = row.locator("[name='new_groep[]'] option", has_text="Anders")
    assert anders.count() == 1


@pytest.mark.usefixtures("login")
def test_confirmed_new_student_submits(live_server, tmp_path, page):
    """A confirmed new student is written to roster.json on forward."""
    proc = _open_roster(live_server, tmp_path, page)
    _add_student(page, "Emma", "Jansen")
    page.click("button:has-text('Wensen via formulier')")
    page.wait_for_url("**/preferences_form")
    roster = json.loads((proc / "roster.json").read_text("utf-8"))
    assert "Emma Jansen" in {
        f"{p['roepnaam']} {p['achternaam']}" for p in roster["participants"]
    }


@pytest.mark.usefixtures("login")
def test_unconfirmed_row_blocks_submit(live_server, tmp_path, page):
    """A started-but-unconfirmed row blocks forward navigation with a message."""
    _open_roster(live_server, tmp_path, page)
    _add_student(page, "Emma", "Jansen", geslacht="", confirm=False)
    page.click("button:has-text('Wensen via formulier')")
    page.wait_for_timeout(300)
    assert "/roster" in page.url
    assert page.locator(".new-student-row .ns-error").first.inner_text() != ""


@pytest.mark.usefixtures("login")
def test_name_collision_flagged_on_confirm(live_server, tmp_path, page):
    """Confirming a new student whose name matches an existing leerling is rejected."""
    _open_roster(live_server, tmp_path, page)
    row = _add_student(page, "Anna", "Bos")  # clashes with candidate s1
    assert row.get_attribute("data-confirmed") == "0"
    assert "bestaat al" in row.locator(".ns-error").inner_text().lower()


@pytest.mark.usefixtures("login")
def test_edit_reopens_chip(live_server, tmp_path, page):
    """'Wijzig' turns a confirmed chip back into an editable row."""
    _open_roster(live_server, tmp_path, page)
    row = _add_student(page, "Emma", "Jansen")
    row.locator("button.ns-edit-btn").click()
    assert row.get_attribute("data-confirmed") == "0"
    assert row.locator(".ns-edit").is_visible()


@pytest.mark.usefixtures("login")
def test_saved_new_student_restored_as_chip(live_server, tmp_path, page):
    """A saved new student comes back as a confirmed chip after a reload."""
    _open_roster(live_server, tmp_path, page)
    _add_student(page, "Emma", "Jansen")
    page.click("button:has-text('Wensen via formulier')")
    page.wait_for_url("**/preferences_form")

    page.goto(f"{live_server}/roster")
    chip = page.locator(".new-student-row .ns-chip")
    assert chip.count() == 1
    assert "Emma Jansen" in chip.first.inner_text()
