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
def test_group_counter_and_empty_marker(live_server, tmp_path, page):
    """Each group shows 'X van Y met voorkeur'; an empty block shows 'nog geen voorkeur'."""
    _open_preferences_form(live_server, tmp_path, page)
    counter = page.locator(
        ".group-details:not(.new-student-details) .group-counter"
    ).first
    assert "0 van 2" in counter.inner_text().lower()
    _open_group(page)
    assert "nog geen voorkeur" in page.locator("#preview-s1").inner_text().lower()

    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    assert "1 van 2" in counter.inner_text().lower()
    assert "nog geen voorkeur" not in page.locator("#preview-s1").inner_text().lower()
    assert "graag met" in page.locator("#preview-s1").inner_text().lower()


@pytest.mark.usefixtures("login")
def test_collapse_group_students_to_preview(live_server, tmp_path, page):
    """'Klap leerlingen dicht' collapses the open candidate blocks in a group."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    assert page.locator("#block-s1").get_attribute("open") is not None
    page.click(".group-details:not(.new-student-details) .group-collapse-btn")
    assert page.locator("#block-s1").get_attribute("open") is None


@pytest.mark.usefixtures("login")
def test_autosave_restores_work_on_reload(live_server, tmp_path, page):
    """A preference is autosaved in the background and survives a reload without saving."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.wait_for_timeout(1500)  # let the debounced autosave reach the server

    page.goto(f"{live_server}/preferences_form")  # reload without an explicit save
    _open_group(page)
    chips = page.locator("#chips-graag_met-s1 .preference-chip")
    assert chips.count() == 1
    assert "Bram Dijk" in chips.first.inner_text()


@pytest.mark.usefixtures("login")
def test_dangling_preference_is_removed_with_undo(live_server, tmp_path, page):
    """Unchecking a student removes preferences pointing to them, with a bundled undo."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_group(page)
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    assert page.locator("#chips-graag_met-s1 .preference-chip").count() == 1

    # Bram (s2) no longer goes over → the preference to Bram is auto-removed.
    page.uncheck("input.gaat-over-checkbox[value='s2']")
    assert page.locator("#chips-graag_met-s1 .preference-chip").count() == 0
    assert page.locator("#undo-banner").is_visible()

    page.click("#undo-banner button:has-text('Ongedaan maken')")
    assert page.locator("#chips-graag_met-s1 .preference-chip").count() == 1


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


def _add_new_student(page, voornaam, achternaam, geslacht="Meisje", confirm=True):
    """Open the new-student section, add one student, optionally click 'Voeg toe'."""
    page.click("details.new-student-details > summary")
    page.click("details.new-student-details button:has-text('Leerling toevoegen')")
    row = page.locator(".new-student-block").last
    row.locator("[name='new_voornaam[]']").fill(voornaam)
    row.locator("[name='new_achternaam[]']").fill(achternaam)
    if geslacht:
        row.locator("[name='new_geslacht[]']").select_option(geslacht)
    if confirm:
        row.locator("button:has-text('Voeg toe')").click()
    return row


@pytest.mark.usefixtures("login")
def test_completed_new_student_is_submitted(live_server, tmp_path, page):
    """A finished ('Voeg toe') student hides its inputs, shows a summary, and is submitted."""
    proc = _open_preferences_form(live_server, tmp_path, page)
    row = _add_new_student(page, "Emma", "Jansen")
    assert row.get_attribute("data-complete") == "1"
    # The input fields disappear; a compact summary (name · sex · origin group) shows.
    assert row.locator(".new-student-edit").is_hidden()
    summary = row.locator(".new-student-summary")
    assert summary.is_visible()
    assert "Uit groep" in summary.inner_text()
    assert summary.locator("button:has-text('Voorkeuren invoeren')").is_visible()

    page.click("button.next-step")
    page.wait_for_url("**/not_together")
    payload = json.loads((proc / "voorkeuren.json").read_text("utf-8"))
    assert "Emma Jansen" in set(payload["student_display"].values())


@pytest.mark.usefixtures("login")
def test_unfinished_new_student_blocks_submit(live_server, tmp_path, page):
    """A started-but-unfinished new student (missing geslacht) blocks submit."""
    _open_preferences_form(live_server, tmp_path, page)
    _add_new_student(page, "Emma", "Jansen", geslacht="", confirm=False)
    page.click("button.next-step")
    # Submit is blocked: we stay on the form and see a message on the row.
    page.wait_for_timeout(300)
    assert "/preferences_form" in page.url
    assert page.locator(".new-student-block .field-error").first.inner_text() != ""


@pytest.mark.usefixtures("login")
def test_new_student_name_collision_is_flagged_on_name_alone(
    live_server, tmp_path, page
):
    """A name clash is flagged live, on the name alone — geslacht need not match."""
    _open_preferences_form(live_server, tmp_path, page)
    # No geslacht filled: the collision must still be detected from the name.
    row = _add_new_student(page, "Anna", "Bos", geslacht="", confirm=False)
    assert (
        "bestaat al"
        in row.locator(".new-student-edit .field-error").inner_text().lower()
    )
    # Trying to finish keeps it uncommitted.
    row.locator("button:has-text('Voeg toe')").click()
    assert row.get_attribute("data-complete") == "0"


@pytest.mark.usefixtures("login")
def test_finished_new_student_is_selectable_as_target(live_server, tmp_path, page):
    """Once finished, a new student can be chosen as a preference of an existing student."""
    _open_preferences_form(live_server, tmp_path, page)
    _add_new_student(page, "Emma", "Jansen")
    _open_group(page)
    page.fill("#combo-graag_met-s1", "Emma")
    assert (
        page.locator(
            "#list-graag_met-s1 .combobox-option:has-text('Emma Jansen')"
        ).count()
        == 1
    )
