# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

"""Browser tests for the preferences_form page JavaScript.

These cover the JS behaviours that the Flask test client cannot exercise: the constrained
combobox (select a known student or group, no free text), chip add/remove, duplicate rules,
tussentijds opslaan round-trip, nieuwe leerling, and forward navigation.
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

    Two destination groups (Klas A/Klas B) so group targets and niet-in are exercised.
    Returns the process directory so tests can inspect written files.
    """
    proc = tmp_path / TEST_SCHOOLCODE / "browsertest"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "relevant_students_and_groups.json").write_text(
        json.dumps({"candidates": CANDIDATES, "groups_from": ["Groep 3"]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"Jongens": [1, 1], "Meisjes": [1, 0]},
        index=pd.Index(["Klas A", "Klas B"], name="Groepen"),
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


def _open_group(page):
    """Click the first real group summary so its candidate blocks become visible."""
    page.click("details.group-details:not(.new-student-details) > summary")


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


@pytest.mark.usefixtures("login")
def test_combobox_opens_on_focus(live_server, tmp_path, page):
    """Focusing a preference combobox opens its option list without typing."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.focus("#combo-graag_met-s1")
    assert page.locator("#list-graag_met-s1 .combobox-option").count() > 0


@pytest.mark.usefixtures("login")
def test_selecting_student_creates_chip_and_keeps_picker_open(
    live_server, tmp_path, page
):
    """Selecting a known classmate adds a chip immediately (no confirm step) and the
    picker stays open for the next entry."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.fill("#combo-graag_met-s1", "Bram")
    page.click("#list-graag_met-s1 .combobox-option:has-text('Bram Dijk')")

    chips = page.locator("#chips-graag_met-s1 .preference-chip")
    assert chips.count() == 1
    assert "Bram Dijk" in chips.first.inner_text()
    # Picker stays open and focused for the next preference.
    assert page.locator("#list-graag_met-s1").is_visible()


@pytest.mark.usefixtures("login")
def test_keyboard_enter_accepts_highlighted_match(live_server, tmp_path, page):
    """Typing a few letters auto-highlights the first match; Enter adds it as a chip."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    chips = page.locator("#chips-graag_met-s1 .preference-chip")
    assert chips.count() == 1
    assert "Bram Dijk" in chips.first.inner_text()


@pytest.mark.usefixtures("login")
def test_keyboard_tab_accepts_highlighted_match(live_server, tmp_path, page):
    """Tab accepts the highlighted match while choosing (like Enter)."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Tab")
    chips = page.locator("#chips-graag_met-s1 .preference-chip")
    assert chips.count() == 1
    assert "Bram Dijk" in chips.first.inner_text()


@pytest.mark.usefixtures("login")
def test_unknown_name_is_not_selectable(live_server, tmp_path, page):
    """Typing an unknown name yields no option, so no chip can be created."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.fill("#combo-graag_met-s1", "Onbekende Persoon")
    assert page.locator("#list-graag_met-s1 .combobox-option").count() == 0
    assert page.locator("#chips-graag_met-s1 .preference-chip").count() == 0


@pytest.mark.usefixtures("login")
def test_selecting_group_creates_chip(live_server, tmp_path, page):
    """A destination group is a valid preference target and becomes a chip."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.fill("#combo-graag_met-s1", "Klas B")
    page.click("#list-graag_met-s1 .combobox-option:has-text('Klas B')")

    chip = page.locator("#chips-graag_met-s1 .preference-chip").first
    assert "Klas B" in chip.inner_text()
    target = page.locator(
        "#chips-graag_met-s1 input[name='preference_s1_graag_met_target']"
    )
    assert target.input_value() == "Klas B"


@pytest.mark.usefixtures("login")
def test_chip_intensity_pills_set_weight(live_server, tmp_path, page):
    """A graag-met chip defaults to weight 1; choosing 'heel graag' sets weight 2."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")

    weight = page.locator(
        "#chips-graag_met-s1 input[name='preference_s1_graag_met_gewicht']"
    )
    assert weight.input_value() == "1"

    page.click("#chips-graag_met-s1 .chip-label")
    page.click("#chips-graag_met-s1 .intensity-pill:has-text('heel graag')")
    assert weight.input_value() == "2"
    assert (
        "heel graag"
        in page.locator("#chips-graag_met-s1 .preference-chip").first.inner_text()
    )


@pytest.mark.usefixtures("login")
def test_liever_niet_has_two_intensity_levels(live_server, tmp_path, page):
    """'Liever niet met' offers two intensity levels (liever niet / echt niet)."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.fill("#combo-liever_niet_met-s1", "Bram")
    page.press("#combo-liever_niet_met-s1", "Enter")
    page.click("#chips-liever_niet_met-s1 .chip-label")
    assert page.locator("#chips-liever_niet_met-s1 .intensity-pill").count() == 2


@pytest.mark.usefixtures("login")
def test_extra_zekerheid_choice_round_trips(live_server, tmp_path, page):
    """'Extra zekerheid' offers fixed choices (default: geen eis); the choice survives
    Tussentijds opslaan and is restored on reload."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.click("#block-s1 .extra-zekerheid > summary")
    geen = page.locator("#block-s1 .extra-zekerheid input[value='']")
    assert geen.is_checked()  # default is no extra requirement

    page.check("#block-s1 .extra-zekerheid input[value='100']")
    page.click("button.button--secondary")
    page.wait_for_url("**/preferences_form")
    _open_group(page)
    page.click("#block-s1 .extra-zekerheid > summary")
    assert page.locator("#block-s1 .extra-zekerheid input[value='100']").is_checked()


@pytest.mark.usefixtures("login")
def test_duplicate_student_is_not_offered_again(live_server, tmp_path, page):
    """Once a classmate is chosen, they are not offered again (uniqueness, ADR 0004)."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.fill("#combo-graag_met-s1", "Bram")
    page.click("#list-graag_met-s1 .combobox-option:has-text('Bram Dijk')")
    # Re-open and look for Bram again — he should be gone from the options.
    page.fill("#combo-graag_met-s1", "Bram")
    assert (
        page.locator(
            "#list-graag_met-s1 .combobox-option:has-text('Bram Dijk')"
        ).count()
        == 0
    )


@pytest.mark.usefixtures("login")
def test_duplicate_group_is_allowed(live_server, tmp_path, page):
    """The same group may be chosen more than once (group preferences stack)."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    for _ in range(2):
        page.fill("#combo-graag_met-s1", "Klas B")
        page.click("#list-graag_met-s1 .combobox-option:has-text('Klas B')")
    assert page.locator("#chips-graag_met-s1 .preference-chip").count() == 2


@pytest.mark.usefixtures("login")
def test_preference_chip_persists_after_opslaan(live_server, tmp_path, page):
    """A chosen preference survives Tussentijds opslaan and is restored on reload."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.fill("#combo-graag_met-s1", "Bram")
    page.click("#list-graag_met-s1 .combobox-option:has-text('Bram Dijk')")

    page.click("button.button--secondary")
    page.wait_for_url("**/preferences_form")
    _open_group(page)

    chips = page.locator("#chips-graag_met-s1 .preference-chip")
    assert chips.count() == 1
    assert "Bram Dijk" in chips.first.inner_text()


@pytest.mark.usefixtures("login")
def test_new_student_row_can_be_added_and_submitted(live_server, tmp_path, page):
    """Adding a new student row and submitting includes them in voorkeuren.json."""
    proc = _open_preferences_form(live_server, tmp_path, page)

    page.click("details.new-student-details > summary")
    page.click("details.new-student-details button:has-text('Leerling toevoegen')")

    row = page.locator(".new-student-block").first
    row.locator("[name='new_voornaam[]']").fill("Emma")
    row.locator("[name='new_achternaam[]']").fill("Jansen")
    row.locator("[name='new_geslacht[]']").select_option("Meisje")

    page.click("button.next-step")
    page.wait_for_url("**/not_together")

    payload = json.loads((proc / "voorkeuren.json").read_text("utf-8"))
    display_names = set(payload["student_display"].values())
    assert "Emma Jansen" in display_names
