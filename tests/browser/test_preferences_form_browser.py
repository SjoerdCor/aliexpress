# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

"""Browser tests for the preferences_form page JavaScript.

These cover the JS behaviours that the Flask test client cannot exercise: the read-only
overview + per-pupil edit modal (open, Opslaan, cancel/revert), the constrained combobox
(select a known student or group, no free text), chip add/remove, duplicate rules, the
read-only row projection (chips, intensity, badge, empty accent), and forward navigation.
The population is fixed by the roster step, so this page only enters preferences.
"""

import json

import pandas as pd
import pytest

from aliexpress.web.extensions import db as flask_db
from aliexpress.web.models import Process
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
    # The roster step ran already (form method chosen), so resume lands on the form.
    (proc / "roster.json").write_text(
        json.dumps({"participants": CANDIDATES}), encoding="utf-8"
    )
    with app.app_context():
        flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name="browsertest"))
        flask_db.session.commit()
    page.goto(f"{live_server}/processes/select/browsertest")
    page.wait_for_url("**/preferences_form")
    return proc


def _open_pupil_modal(page, key):
    """Click a pupil's read-only row to open that pupil's edit modal."""
    page.click(f".candidate-row[data-key='{key}']")


# ── tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("login")
def test_clicking_pupil_row_opens_edit_modal(live_server, tmp_path, page):
    """Clicking a pupil's read-only row opens the edit modal for that pupil, titled with
    the pupil's full name. The editor is hidden until the row is clicked."""
    _open_preferences_form(live_server, tmp_path, page)
    editor = page.locator("#editor-s1")
    assert not editor.is_visible()  # hidden until the row is clicked
    _open_pupil_modal(page, "s1")
    assert editor.is_visible()
    assert "Anna Bos" in page.locator("#editor-s1 .modal-title").inner_text()


@pytest.mark.usefixtures("login")
def test_modal_opslaan_closes_and_persists(live_server, tmp_path, page):
    """The modal's only exit, 'Opslaan', closes the modal and saves in the background; a
    reload (without an explicit form save) restores the entered preference."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.click("#editor-s1 .modal-done")
    assert not page.locator("#editor-s1").is_visible()
    assert not page.locator("#pref-scrim").is_visible()

    page.goto(f"{live_server}/preferences_form")  # reload without an explicit save
    _open_pupil_modal(page, "s1")
    chips = page.locator("#chips-graag_met-s1 .preference-chip")
    assert chips.count() == 1
    assert "Bram" in chips.first.inner_text()


@pytest.mark.usefixtures("login")
def test_esc_does_not_close_modal(live_server, tmp_path, page):
    """Esc no longer closes the modal: the edit stays put so a stray keypress cannot lose
    work — only the explicit Annuleren button discards it."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.keyboard.press("Escape")
    assert page.locator("#editor-s1").is_visible()  # still open
    assert page.locator("#chips-graag_met-s1 .preference-chip").count() == 1  # kept


@pytest.mark.usefixtures("login")
def test_annuleren_discards_and_backdrop_does_not_close(live_server, tmp_path, page):
    """Annuleren closes the modal and discards the changes; a backdrop click does not close
    it, so work cannot be lost by an accidental click outside the modal."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.click("#editor-s1 .modal-cancel")
    assert not page.locator("#editor-s1").is_visible()
    assert page.locator("#rowchips-s1 .chip").count() == 0  # discarded

    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.mouse.click(5, 5)  # a corner only the scrim covers
    assert page.locator("#editor-s1").is_visible()  # backdrop does not close
    assert page.locator("#chips-graag_met-s1 .preference-chip").count() == 1  # kept


@pytest.mark.usefixtures("login")
def test_changes_persist_only_after_explicit_opslaan(live_server, tmp_path, page):
    """After cancelling pupil A with Annuleren, saving pupil B must not also persist A's
    discarded edit (the whole-form save would otherwise leak it)."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.click("#editor-s1 .modal-cancel")  # discard A

    _open_pupil_modal(page, "s2")
    page.fill("#combo-graag_met-s2", "Anna")
    page.press("#combo-graag_met-s2", "Enter")
    page.click("#editor-s2 .modal-done")  # save B

    page.goto(f"{live_server}/preferences_form")  # reload from the saved draft
    assert page.locator("#rowchips-s1 .chip").count() == 0  # A's edit did not leak
    assert page.locator("#rowchips-s2 .chip").count() == 1


@pytest.mark.usefixtures("login")
def test_focus_moves_to_title_on_open_and_back_to_row_on_close(
    live_server, tmp_path, page
):
    """Opening focuses the modal title (not the first field, which would pop a combobox);
    closing returns focus to the row that was clicked."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    assert "modal-title" in (
        page.evaluate("document.activeElement && document.activeElement.className")
        or ""
    )
    page.click("#editor-s1 .modal-done")
    assert (
        page.evaluate("document.activeElement && document.activeElement.id") == "row-s1"
    )


@pytest.mark.usefixtures("login")
def test_row_projects_readonly_chips_after_save(live_server, tmp_path, page):
    """After saving, the read-only row shows a chip per preference (a projection of the
    editor's chips), carrying the kind class for the visual language."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.click("#editor-s1 .modal-done")

    row_chips = page.locator("#rowchips-s1 .chip")
    assert row_chips.count() == 1
    assert page.locator("#rowchips-s1 .chip--graag").count() == 1
    assert "Bram" in row_chips.first.inner_text()


@pytest.mark.usefixtures("login")
def test_row_chip_encodes_intensity_glyph_and_saturation(live_server, tmp_path, page):
    """Weight maps to a trailing glyph and a saturation class on the read-only chip:
    'heel graag' → ↑ + w-strong, 'hartsvriend' → ♥ + w-top."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.click("#chips-graag_met-s1 .chip-label")
    page.click("#chips-graag_met-s1 .intensity-pill:has-text('heel graag')")
    page.click("#editor-s1 .modal-done")

    chip = page.locator("#rowchips-s1 .chip--graag")
    assert "w-strong" in (chip.get_attribute("class") or "")
    assert "↑" in chip.inner_text()

    _open_pupil_modal(page, "s1")
    page.click("#chips-graag_met-s1 .chip-label")
    page.click("#chips-graag_met-s1 .intensity-pill:has-text('hartsvriend')")
    page.click("#editor-s1 .modal-done")
    chip = page.locator("#rowchips-s1 .chip--graag")
    assert "w-top" in (chip.get_attribute("class") or "")
    assert "♥" in chip.inner_text()


@pytest.mark.usefixtures("login")
def test_row_shows_extra_zekerheid_badge_when_set(live_server, tmp_path, page):
    """Setting an extra-zekerheid requirement shows a badge next to the pupil's name."""
    _open_preferences_form(live_server, tmp_path, page)
    assert page.locator("#rowbadge-s1 .badge-zeker").count() == 0  # none by default
    _open_pupil_modal(page, "s1")
    page.click("#editor-s1 .extra-zekerheid > summary")
    page.check("#editor-s1 .extra-zekerheid input[value='100']")
    page.click("#editor-s1 .modal-done")
    assert page.locator("#rowbadge-s1 .badge-zeker").count() == 1


@pytest.mark.usefixtures("login")
def test_badge_distinguishes_two_extra_zekerheid_levels(live_server, tmp_path, page):
    """The badge differs between 'minstens tevreden' (partial) and 'alle voorkeuren
    gehonoreerd' (full)."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.click("#editor-s1 .extra-zekerheid > summary")
    page.check("#editor-s1 .extra-zekerheid input[value='50']")
    page.click("#editor-s1 .modal-done")
    assert page.locator("#rowbadge-s1 .badge-zeker--partial").count() == 1

    _open_pupil_modal(page, "s1")
    page.check("#editor-s1 .extra-zekerheid input[value='100']")
    page.click("#editor-s1 .modal-done")
    assert page.locator("#rowbadge-s1 .badge-zeker--full").count() == 1


@pytest.mark.usefixtures("login")
def test_row_with_nothing_set_shows_neutral_placeholder(live_server, tmp_path, page):
    """A pupil with nothing set at all gets a neutral 'nog niet ingevuld' accent — not a
    warning. Adding a preference clears it."""
    _open_preferences_form(live_server, tmp_path, page)
    row = page.locator("#row-s2")
    assert "candidate-row--empty" in (row.get_attribute("class") or "")
    assert "nog niet ingevuld" in page.locator("#rowchips-s2").inner_text().lower()

    _open_pupil_modal(page, "s2")
    page.fill("#combo-graag_met-s2", "Anna")
    page.press("#combo-graag_met-s2", "Enter")
    page.click("#editor-s2 .modal-done")
    assert "candidate-row--empty" not in (
        page.locator("#row-s2").get_attribute("class") or ""
    )


@pytest.mark.usefixtures("login")
def test_row_uses_short_name_but_stores_full_target(live_server, tmp_path, page):
    """The overview shows short unique names, while the stored preference target keeps the
    full name (display-only short names, ADR 0007)."""
    _open_preferences_form(live_server, tmp_path, page)
    assert page.locator("#row-s1 .row-name").inner_text().strip() == "Anna"

    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    target = page.locator(
        "#chips-graag_met-s1 input[name='preference_s1_graag_met_target']"
    )
    assert target.input_value() == "Bram Dijk"  # full name stored
    page.click("#editor-s1 .modal-done")

    row_chip = page.locator("#rowchips-s1 .chip--graag")
    assert "Bram" in row_chip.inner_text()
    assert "Dijk" not in row_chip.inner_text()  # short name shown


@pytest.mark.usefixtures("login")
def test_no_tussentijds_opslaan_button(live_server, tmp_path, page):
    """The bottom 'Tussentijds opslaan' button is gone: each modal's 'Opslaan' persists,
    so a separate page-wide save is redundant (ADR 0007)."""
    _open_preferences_form(live_server, tmp_path, page)
    assert page.locator("button[value='opslaan']").count() == 0


@pytest.mark.usefixtures("login")
def test_intensity_pills_visible_by_default(live_server, tmp_path, page):
    """The intensity pills are shown straight away on a fresh chip — the teacher picks the
    weight without an extra click to reveal them."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    pills = page.locator("#chips-graag_met-s1 .chip-pills")
    assert pills.is_visible()


@pytest.mark.usefixtures("login")
def test_readonly_chip_has_descriptive_hover_title(live_server, tmp_path, page):
    """A read-only chip carries a hover title naming the kind and intensity, so a teacher
    can tell what it means without opening the modal."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.click("#chips-graag_met-s1 .chip-label")
    page.click("#chips-graag_met-s1 .intensity-pill:has-text('heel graag')")
    page.click("#editor-s1 .modal-done")
    title = page.locator("#rowchips-s1 .chip--graag").get_attribute("title")
    assert "Graag met" in title
    assert "heel graag" in title


@pytest.mark.usefixtures("login")
def test_row_chips_sorted_by_weight_within_type(live_server, tmp_path, page):
    """Within a type, the row shows the heaviest wish first regardless of entry order."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    # Enter a normal-weight group first, then a hartsvriend classmate (heavier, entered later).
    page.fill("#combo-graag_met-s1", "Klas A")
    page.press("#combo-graag_met-s1", "Enter")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.click("#chips-graag_met-s1 .preference-chip:has-text('Bram') .chip-label")
    page.click(
        "#chips-graag_met-s1 .preference-chip:has-text('Bram') "
        ".intensity-pill:has-text('hartsvriend')"
    )
    page.click("#editor-s1 .modal-done")
    chips = page.locator("#rowchips-s1 .chip--graag")
    assert "Bram" in chips.nth(0).inner_text()  # weight 5 first
    assert "Klas A" in chips.nth(1).inner_text()  # weight 1 second


@pytest.mark.usefixtures("login")
def test_group_counter_counts_pupils_with_preference(live_server, tmp_path, page):
    """The flat class header shows 'X van Y met voorkeur', updated as preferences are added."""
    _open_preferences_form(live_server, tmp_path, page)
    counter = page.locator(".group-counter").first
    assert "0 van 2" in counter.inner_text().lower()

    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.click("#editor-s1 .modal-done")
    assert "1 van 2" in counter.inner_text().lower()


@pytest.mark.usefixtures("login")
def test_forward_button_navigates_to_not_together(live_server, tmp_path, page):
    """'Naar niet samen →' submits the form and redirects to /not_together."""
    _open_preferences_form(live_server, tmp_path, page)
    page.click("button.next-step")
    page.wait_for_url("**/not_together")


@pytest.mark.usefixtures("login")
def test_help_is_collapsible_and_open_by_default(live_server, tmp_path, page):
    """The page help is a collapsible block, open by default, with a worked example."""
    _open_preferences_form(live_server, tmp_path, page)
    help_box = page.locator("details.instructions-box")
    assert help_box.get_attribute("open") is not None
    assert "voorbeeld" in help_box.inner_text().lower()


@pytest.mark.usefixtures("login")
def test_chip_visual_class_per_kind(live_server, tmp_path, page):
    """Graag-met and liever-niet chips carry distinct kind classes for the visual language."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    page.fill("#combo-liever_niet_met-s1", "Klas B")
    page.press("#combo-liever_niet_met-s1", "Enter")
    assert page.locator("#chips-graag_met-s1 .preference-chip--graag_met").count() == 1
    assert (
        page.locator(
            "#chips-liever_niet_met-s1 .preference-chip--liever_niet_met"
        ).count()
        == 1
    )


@pytest.mark.usefixtures("login")
def test_info_popover_toggles(live_server, tmp_path, page):
    """Clicking a heavy ⓘ opens a popover; clicking it again closes it."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    info = page.locator("#editor-s1 .preference-section .info-pop").first
    info.click()
    assert page.locator(".info-popover").count() == 1
    info.click()
    assert page.locator(".info-popover").count() == 0


@pytest.mark.usefixtures("login")
def test_combobox_opens_on_focus(live_server, tmp_path, page):
    """Focusing a preference combobox opens its option list without typing."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.focus("#combo-graag_met-s1")
    assert page.locator("#list-graag_met-s1 .combobox-option").count() > 0


@pytest.mark.usefixtures("login")
def test_selecting_student_creates_chip_and_keeps_picker_open(
    live_server, tmp_path, page
):
    """Selecting a known classmate adds a chip immediately (no confirm step) and the
    picker stays open for the next entry."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.click("#list-graag_met-s1 .combobox-option:has-text('Bram')")

    chips = page.locator("#chips-graag_met-s1 .preference-chip")
    assert chips.count() == 1
    assert "Bram" in chips.first.inner_text()
    # Picker stays open and focused for the next preference.
    assert page.locator("#list-graag_met-s1").is_visible()


@pytest.mark.usefixtures("login")
def test_keyboard_enter_accepts_highlighted_match(live_server, tmp_path, page):
    """Typing a few letters auto-highlights the first match; Enter adds it as a chip."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Enter")
    chips = page.locator("#chips-graag_met-s1 .preference-chip")
    assert chips.count() == 1
    assert "Bram" in chips.first.inner_text()


@pytest.mark.usefixtures("login")
def test_keyboard_tab_accepts_highlighted_match(live_server, tmp_path, page):
    """Tab accepts the highlighted match while choosing (like Enter)."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.press("#combo-graag_met-s1", "Tab")
    chips = page.locator("#chips-graag_met-s1 .preference-chip")
    assert chips.count() == 1
    assert "Bram" in chips.first.inner_text()


@pytest.mark.usefixtures("login")
def test_unknown_name_is_not_selectable(live_server, tmp_path, page):
    """Typing an unknown name yields no option, so no chip can be created."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Onbekende Persoon")
    assert page.locator("#list-graag_met-s1 .combobox-option").count() == 0
    assert page.locator("#chips-graag_met-s1 .preference-chip").count() == 0


@pytest.mark.usefixtures("login")
def test_selecting_group_creates_chip(live_server, tmp_path, page):
    """A destination group is a valid preference target and becomes a chip."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
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
    _open_pupil_modal(page, "s1")
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
    _open_pupil_modal(page, "s1")
    page.fill("#combo-liever_niet_met-s1", "Bram")
    page.press("#combo-liever_niet_met-s1", "Enter")
    page.click("#chips-liever_niet_met-s1 .chip-label")
    assert page.locator("#chips-liever_niet_met-s1 .intensity-pill").count() == 2


@pytest.mark.usefixtures("login")
def test_extra_zekerheid_choice_round_trips(live_server, tmp_path, page):
    """'Extra zekerheid' offers fixed choices (default: geen eis); the choice survives the
    modal's Opslaan and a reload."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    # Extra zekerheid is collapsed by default; open it to reveal the choices + help text.
    page.click("#editor-s1 .extra-zekerheid > summary")
    geen = page.locator("#editor-s1 .extra-zekerheid input[value='']")
    assert geen.is_checked()  # default is no extra requirement

    page.check("#editor-s1 .extra-zekerheid input[value='100']")
    page.click("#editor-s1 .modal-done")

    page.goto(f"{live_server}/preferences_form")  # reload from the saved draft
    _open_pupil_modal(page, "s1")
    page.click("#editor-s1 .extra-zekerheid > summary")
    assert page.locator("#editor-s1 .extra-zekerheid input[value='100']").is_checked()


@pytest.mark.usefixtures("login")
def test_duplicate_student_is_not_offered_again(live_server, tmp_path, page):
    """Once a classmate is chosen, they are not offered again (uniqueness, ADR 0004)."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    page.fill("#combo-graag_met-s1", "Bram")
    page.click("#list-graag_met-s1 .combobox-option:has-text('Bram')")
    # Re-open and look for Bram again — he should be gone from the options.
    page.fill("#combo-graag_met-s1", "Bram")
    assert (
        page.locator("#list-graag_met-s1 .combobox-option:has-text('Bram')").count()
        == 0
    )


@pytest.mark.usefixtures("login")
def test_duplicate_group_is_allowed(live_server, tmp_path, page):
    """The same group may be chosen more than once (group preferences stack)."""
    _open_preferences_form(live_server, tmp_path, page)
    _open_pupil_modal(page, "s1")
    for _ in range(2):
        page.fill("#combo-graag_met-s1", "Klas B")
        page.click("#list-graag_met-s1 .combobox-option:has-text('Klas B')")
    assert page.locator("#chips-graag_met-s1 .preference-chip").count() == 2
