# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

"""Browser tests for the groups-to page JavaScript (enable/disable, adding groups)."""

import json

import pandas as pd
import pytest


def student(geslacht, roepnaam="Test", *, blijft=True):
    """Build one groups-to student dict as the candidates JSON stores them."""
    return {
        "geslacht": geslacht,
        "roepnaam": roepnaam,
        "achternaam": "Leerling",
        "jaargroep": 4,
        "blijft_in_groep": blijft,
    }


def _state(proc):
    return json.loads((proc / "groups_to_state.json").read_text("utf-8"))


def _groups_xlsx(proc):
    return pd.read_excel(proc / "groups.xlsx", index_col=0)


def test_switched_off_group_keeps_its_ticks(open_groups_to, page):
    """A disabled group keeps its successful student controls and restores their ticks."""
    proc = open_groups_to(
        {
            "Klas A": [student("Jongen")],
            "Klas B": [student("Meisje")],  # ticked by default (blijft_in_groep)
            "Klas C": [student("Jongen")],
        }
    )
    group = page.locator('.groups-to-group[data-group="Klas B"]')
    group.locator("[data-group-toggle]").uncheck()
    assert group.locator(".group-students").get_attribute("hidden") is not None
    assert group.locator(".student-list input").is_checked()
    assert not group.locator(".student-list input").is_disabled()
    group.locator("[data-group-toggle]").check()
    assert group.locator(".group-students").get_attribute("hidden") is None
    assert group.locator(".student-list input").is_checked()
    group.locator("[data-group-toggle]").uncheck()
    page.click("button:has-text('Voorkeuren invullen via Excel')")
    page.wait_for_url("**/preferences_excel")

    state = _state(proc)
    assert state["disabled_groups"] == ["Klas B"]
    # Its checkbox stayed enabled, so the tick is remembered for when it is switched on.
    assert state["original_groups"]["Klas B"]["checked_indices"] == [0]
    assert "Klas B" not in _groups_xlsx(proc).index

    page.goto(page.url.replace("/preferences_excel", "/groups_to"))
    restored = page.locator('.groups-to-group[data-group="Klas B"]')
    assert not restored.locator("[data-group-toggle]").is_checked()
    restored.locator("[data-group-toggle]").check()
    assert restored.locator(".student-list input").is_checked()


def test_added_empty_group_is_saved(open_groups_to, page):
    """A new empty group can be renamed, removed, and stored at 0/0."""
    proc = open_groups_to(
        {"Klas A": [student("Jongen")], "Klas B": [student("Meisje")]}
    )
    assert page.locator(".groups-to-add").get_attribute("open") is None
    page.locator(".groups-to-add > summary").click()
    page.click('button:has-text("Lege groep toevoegen")')
    first = page.locator("#new-groups input.group-name-input").last
    assert first.input_value() == "Nieuwe groep 1"
    assert first.evaluate("element => document.activeElement === element")
    first.fill("Extra groep")
    page.click('button:has-text("Lege groep toevoegen")')
    second = page.locator("#new-groups input.group-name-input").last
    assert second.input_value() == "Nieuwe groep 1"
    second.fill("Te verwijderen groep")
    second.locator("xpath=../following-sibling::button").click()
    assert page.locator("#new-groups input.group-name-input").count() == 1
    page.click("button:has-text('Voorkeuren invullen via Excel')")
    page.wait_for_url("**/preferences_excel")

    assert _state(proc)["new_groups"] == ["Extra groep"]
    saved = _groups_xlsx(proc)
    assert saved.loc["Extra groep", "Jongens"] == 0
    assert saved.loc["Extra groep", "Meisjes"] == 0


def test_counts_update_with_correct_singular_forms(open_groups_to, page):
    """The server count is present immediately and one live region updates in place."""
    open_groups_to(
        {
            "Klas A": [student("Jongen"), student("Meisje")],
            "Klas B": [student("Meisje", roepnaam="Lange")],
        }
    )
    count = page.locator('.groups-to-group[data-group="Klas A"] .group-count')
    assert (
        count.inner_text()
        == "Blijven in deze groep: 2 leerlingen · 1 jongen · 1 meisje"
    )
    assert (
        page.locator(
            '.groups-to-group[data-group="Klas A"] [aria-live="polite"]'
        ).count()
        == 1
    )
    page.locator(
        '.groups-to-group[data-group="Klas A"] .student-list input'
    ).first.uncheck()
    assert (
        count.inner_text() == "Blijven in deze groep: 1 leerling · 0 jongens · 1 meisje"
    )


def test_keyboard_validation_and_group_controls(open_groups_to, page):
    """The group toggle, add action, and all client validations work without a mouse."""
    open_groups_to(
        {
            "Klas A": [student("Jongen")],
            "Klas B": [student("Meisje")],
        }
    )
    toggle = page.locator('.groups-to-group[data-group="Klas A"] [data-group-toggle]')
    toggle.focus()
    page.keyboard.press("Space")
    assert toggle.is_checked() is False
    assert page.locator("#groups-to-client-message").is_hidden()

    page.locator('.groups-to-group[data-group="Klas B"] [data-group-toggle]').uncheck()
    page.click("button:has-text('Voorkeuren invullen via Excel')")
    error = page.locator("#groups-to-client-message")
    error_text = error.locator(".groups-to-client-message-text")
    assert error_text.inner_text() == "Kies minimaal twee groepen voor volgend jaar."
    assert error.evaluate("element => document.activeElement === element")

    page.locator('.groups-to-group[data-group="Klas A"] [data-group-toggle]').check()

    page.locator(".groups-to-add > summary").click()
    add = page.locator("#add-empty-group")
    add.focus()
    page.keyboard.press("Enter")
    assert page.locator("#new-groups input.group-name-input").last.evaluate(
        "element => document.activeElement === element"
    )

    page.locator("#new-groups input.group-name-input").last.fill("")
    page.click("button:has-text('Voorkeuren invullen via Excel')")
    assert error_text.inner_text() == "Geef iedere nieuwe groep een naam."
    assert error.is_visible()
    assert error.get_attribute("role") == "alert"
    assert error.evaluate("element => document.activeElement === element")

    page.locator("#new-groups input.group-name-input").last.fill("Klas A")
    page.click("button:has-text('Voorkeuren invullen via Excel')")
    assert error_text.inner_text() == (
        "Iedere groep heeft een unieke naam nodig. Pas de dubbele groepsnaam ‘Klas A’ aan."
    )
    assert error.evaluate("element => document.activeElement === element")


def test_server_validation_redirect_restores_draft_and_focuses_error(
    open_groups_to, page
):
    """A server-side validation redirect keeps the draft and focuses its error."""
    open_groups_to(
        {
            "Klas A": [student("Jongen")],
            "Klas B": [student("Meisje")],
        }
    )
    page.locator(".groups-to-add > summary").click()
    page.click('button:has-text("Lege groep toevoegen")')
    page.locator("#new-groups input.group-name-input").last.fill("")

    with page.expect_navigation():
        page.locator("#groups-to-form").evaluate("form => form.submit()")

    error = page.locator("body > .flash-message.error")
    assert error.inner_text().strip().endswith("Geef iedere nieuwe groep een naam.")
    assert error.get_attribute("role") == "alert"
    assert error.evaluate("element => document.activeElement === element")
    assert page.locator("#new-groups input.group-name-input").last.input_value() == ""


@pytest.mark.parametrize("width", [1280, 390, 320])
@pytest.mark.parametrize("zoom", [1, 2])
def test_groups_to_fits_narrow_viewports_and_zoom(open_groups_to, page, width, zoom):
    """Long names remain reachable without horizontal overflow at supported sizes."""
    long_group = (
        "Een uitzonderlijk lange groepsnaam die moet kunnen omlopen zonder afkappen"
    )
    long_student = "Een leerling met een uitzonderlijk lange achternaam voor controle"
    open_groups_to(
        {
            long_group: [student("Meisje", roepnaam=long_student)],
            "Klas blauw": [student("Jongen")],
        }
    )
    page.set_viewport_size({"width": width, "height": 900})
    if zoom != 1:
        page.evaluate("zoom => { document.documentElement.style.zoom = zoom; }", zoom)
    page.locator("[data-group-toggle]").first.focus()
    overflow = page.evaluate(
        """() => ({width: innerWidth, scroll: document.documentElement.scrollWidth,
        offenders: [...document.querySelectorAll('body *')].filter(e => {
          const r = e.getBoundingClientRect();
          return r.width > 0 && r.right > innerWidth + 1;
        }).slice(0, 5).map(e => ({tag:e.tagName, cls:e.className, id:e.id}))})"""
    )
    assert overflow["scroll"] <= overflow["width"], overflow
    assert not overflow["offenders"], overflow
