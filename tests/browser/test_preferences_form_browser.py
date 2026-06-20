# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

"""Browser tests for the preferences_form page JavaScript.

These cover the JS behaviours that the Flask test client cannot exercise:
chip add/confirm/cancel, tussentijds opslaan flash, nieuwe leerling, forward navigation.
"""

import json

import pandas as pd
import pytest

from aliexpress.extensions import db as flask_db
from aliexpress.models import Process
from app import app
from tests.browser.conftest import TEST_SCHOOLCODE

# ── helpers ──────────────────────────────────────────────────────────────────

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


def _open_preferences_form(live_server, tmp_path, page):
    """Set up a process and navigate the browser to /preferences_form.

    Returns the process directory so tests can inspect written files.
    """
    proc = tmp_path / TEST_SCHOOLCODE / "browsertest"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "relevant_students_and_groups.json").write_text(
        json.dumps({"candidates": CANDIDATES, "groups_from": ["Groep 3"]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"Jongens": [1], "Meisjes": [1]},
        index=pd.Index(["Klas A"], name="Groepen"),
    ).to_excel(proc / "groups.xlsx")
    (proc / "input_method.json").write_text(
        json.dumps({"method": "form"}), encoding="utf-8"
    )
    with app.app_context():
        flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name="browsertest"))
        flask_db.session.commit()
    page.goto(f"{live_server}/processes/select/browsertest")
    page.wait_for_url("**/preferences_form")
    return proc


# ── tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("login")
def test_forward_button_navigates_to_not_together(live_server, tmp_path, page):
    """'Naar niet samen →' submits the form and redirects to /not_together."""
    _open_preferences_form(live_server, tmp_path, page)
    page.click("button.next-step")
    page.wait_for_url("**/not_together")


@pytest.mark.usefixtures("login")
def test_tussentijds_opslaan_shows_flash_and_stays_on_page(live_server, tmp_path, page):
    """'Tussentijds opslaan' posts the form, flashes a success message, and stays on
    /preferences_form."""
    _open_preferences_form(live_server, tmp_path, page)
    page.click("button.button--secondary")
    page.wait_for_url("**/preferences_form")
    flash = page.locator(".flash-message")
    assert flash.count() > 0
    assert "opgeslagen" in flash.first.inner_text().lower()


def _open_group(page):
    """Click the first real group summary so its candidate blocks become visible."""
    page.click("details.group-details:not(.new-student-details) > summary")


@pytest.mark.usefixtures("login")
def test_wish_chip_appears_after_confirm(live_server, tmp_path, page):
    """Adding a wish for Anna (student s1): fill name + weight, confirm → chip visible."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)

    # Click "+ Wens toevoegen" for Graag met
    page.click("#btn-graag_met-s1")

    # Fill in the wish
    pending = page.locator("#pending-graag_met-s1")
    pending.locator(".wish-name-input").fill("Bram Dijk")
    pending.locator(".wish-weight-input").fill("2")
    page.click("#pending-graag_met-s1 button:has-text('Bevestig')")

    # Chip must appear
    chips = page.locator("#chips-graag_met-s1 .wish-chip")
    assert chips.count() == 1
    assert "Bram Dijk" in chips.first.inner_text()


@pytest.mark.usefixtures("login")
def test_wish_chip_persists_after_opslaan(live_server, tmp_path, page):
    """A confirmed wish chip is still visible after Tussentijds opslaan reloads the page."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)

    # Add wish
    page.click("#btn-graag_met-s1")
    page.locator("#pending-graag_met-s1 .wish-name-input").fill("Bram Dijk")
    page.locator("#pending-graag_met-s1 .wish-weight-input").fill("1")
    page.click("#pending-graag_met-s1 button:has-text('Bevestig')")

    # Tussentijds opslaan
    page.click("button.button--secondary")
    page.wait_for_url("**/preferences_form")
    _open_group(page)

    # Chip must be restored from state — proves the wish was persisted and reloaded
    chips = page.locator("#chips-graag_met-s1 .wish-chip")
    assert chips.count() == 1
    assert "Bram Dijk" in chips.first.inner_text()


@pytest.mark.usefixtures("login")
def test_unknown_name_shows_validation_error(live_server, tmp_path, page):
    """Confirming a wish for an unknown name shows a field-error, no chip is added."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)

    page.click("#btn-graag_met-s1")
    page.locator("#pending-graag_met-s1 .wish-name-input").fill("Onbekende Persoon")
    page.click("#pending-graag_met-s1 button:has-text('Bevestig')")

    assert page.locator("#err-graag_met-s1").inner_text() != ""
    assert page.locator("#chips-graag_met-s1 .wish-chip").count() == 0


@pytest.mark.usefixtures("login")
def test_new_student_row_can_be_added_and_submitted(live_server, tmp_path, page):
    """Adding a new student row and submitting includes them in voorkeuren.json."""
    proc = _open_preferences_form(live_server, tmp_path, page)

    # Open the "Nieuwe leerling toevoegen" details
    page.click("details.new-student-details > summary")
    page.click("details.new-student-details button:has-text('Leerling toevoegen')")

    # Fill in the new student
    row = page.locator(".new-student-block").first
    row.locator("[name='new_voornaam[]']").fill("Emma")
    row.locator("[name='new_achternaam[]']").fill("Jansen")
    row.locator("[name='new_geslacht[]']").select_option("Meisje")

    # Submit
    page.click("button.next-step")
    page.wait_for_url("**/not_together")

    payload = json.loads((proc / "voorkeuren.json").read_text("utf-8"))
    display_names = set(payload["student_display"].values())
    assert "Emma Jansen" in display_names
