"""Browser acceptance checks for the EDEXML upload step."""

import pytest
from playwright.sync_api import expect


def _create_process(live_server, page, mode="forward", name="upload-browser-test"):
    """Create a process through the normal process form and open the upload step."""
    page.goto(f"{live_server}/processes?new=1")
    page.fill("#processName", name)
    if mode != "forward":
        page.check(f"input[name='mode'][value='{mode}']")
    page.click("#processForm button[type=submit]")
    page.wait_for_url(f"{live_server}/upload_edexml")


def _assert_no_horizontal_overflow(page):
    """The page must fit the viewport without a document-level horizontal scrollbar."""
    overflow = page.evaluate(
        """() => ({width: innerWidth, scroll: document.documentElement.scrollWidth,
        offenders: [...document.querySelectorAll('body *')].filter(e => {
          const r = e.getBoundingClientRect();
          return r.width > 0 && r.right > innerWidth + 1;
        }).slice(0, 8).map(e => ({tag:e.tagName, cls:e.className, id:e.id}))})"""
    )
    assert overflow["scroll"] <= overflow["width"], overflow


def _assert_help_details_is_keyboard_operable(page):
    """The collapsed help disclosure opens and closes with the keyboard."""
    details = page.locator("details.upload-edexml-help-details")
    summary = details.locator("summary")
    expect(details).not_to_have_attribute("open")
    summary.focus()
    expect(summary).to_be_focused()
    page.keyboard.press("Enter")
    expect(details).to_have_attribute("open", "")
    page.keyboard.press("Enter")
    expect(details).not_to_have_attribute("open")


@pytest.mark.usefixtures("login")
@pytest.mark.parametrize("width", [1280, 390, 320])
def test_forward_upload_page_is_responsive_and_keyboard_operable(
    live_server, page, width
):
    """The forward page keeps its controls named, reachable and visible when narrow."""
    page.set_viewport_size({"width": width, "height": 720})
    _create_process(live_server, page, name=f"upload-forward-{width}")

    expect(page.locator("h1")).to_have_text("Leerlinggegevens ophalen")
    file_input = page.get_by_label("EDEXML-bestand", exact=True)
    year_select = page.get_by_label("Huidige jaarlaag")
    submit = page.get_by_role(
        "button", name="Gegevens inlezen en leerlingen controleren →"
    )
    expect(file_input).to_have_attribute("accept", ".xml")
    expect(year_select).to_be_visible()
    expect(submit).to_be_visible()
    expect(page.locator("legend")).to_have_text(
        "Welke huidige jaarlaag wil je indelen?"
    )
    expect(page.locator("#jaargroep-help")).to_have_text(
        "Kies de jaarlaag van de leerlingen die volgend schooljaar naar de volgende groepen gaan."
    )

    for control in (file_input, year_select, submit):
        control.focus()
        expect(control).to_be_focused()

    _assert_help_details_is_keyboard_operable(page)
    page.locator("details.upload-edexml-help-details summary").focus()
    page.keyboard.press("Enter")
    link = page.get_by_text(
        "Zoek op hoe je een EDEXML-bestand exporteert ↗", exact=True
    )
    expect(link).to_be_visible()
    expect(link).to_have_attribute("target", "_blank")
    expect(link).to_have_attribute("rel", "noopener noreferrer")
    expect(link).to_have_attribute(
        "aria-label",
        "Zoek op hoe je een EDEXML-bestand exporteert ↗ "
        "(opent een externe zoekopdracht)",
    )
    _assert_no_horizontal_overflow(page)


@pytest.mark.usefixtures("login")
@pytest.mark.parametrize("mode", ["redistribute", "redistribute_and_forward"])
def test_redistribute_upload_modes_show_their_approved_controls(
    live_server, page, mode
):
    """Each redistribution mode renders only its approved year-layer controls and action."""
    page.set_viewport_size({"width": 390, "height": 720})
    _create_process(live_server, page, mode=mode, name=f"upload-{mode}-browser-test")

    expect(page.locator("h1")).to_have_text("Leerlinggegevens ophalen")
    expect(page.get_by_label("EDEXML-bestand", exact=True)).to_be_visible()
    expect(page.locator("select[name=jaargroep]")).to_have_count(0)
    if mode == "redistribute":
        expect(page.locator("input[name=jaargroepen]")).to_have_count(0)
        expect(
            page.get_by_role("button", name="Gegevens inlezen en groepen kiezen →")
        ).to_be_visible()
        expect(page.locator("fieldset")).to_have_count(0)
    else:
        expect(page.locator("input[name=jaargroepen]")).to_have_count(8)
        expect(page.locator("legend")).to_have_text(
            "Welke huidige jaarlagen wil je indelen?"
        )
        expect(page.locator("#jaargroepen-help")).to_have_text(
            "Selecteer alle jaarlagen die een jaar verder gaan en daarbij opnieuw over de "
            "groepen worden verdeeld."
        )
        expect(
            page.get_by_role(
                "button", name="Gegevens inlezen en leerlingen controleren →"
            )
        ).to_be_visible()
        for year in range(1, 9):
            expect(page.locator(f"label[for='jaargroep-{year}']")).to_be_visible()

    _assert_help_details_is_keyboard_operable(page)
    _assert_no_horizontal_overflow(page)
