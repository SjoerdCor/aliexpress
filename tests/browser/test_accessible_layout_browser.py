"""Focused responsive and accessibility checks for the processing page."""

import pandas as pd
import pytest
from playwright.sync_api import expect

from aliexpress.data.preferences_form import StudentEntry
from tests.browser.test_distribution_browser import _make_process
from tests.helpers import write_minimal_voorkeuren_json


def _has_horizontal_overflow(page):
    return page.evaluate(
        """() => Math.max(document.documentElement.scrollWidth, document.body.scrollWidth)
        <= document.documentElement.clientWidth"""
    )


def _write_long_processing_input(proc):
    """Replace the small fixture names with long, fictitious review names."""
    write_minimal_voorkeuren_json(
        proc,
        students=[
            StudentEntry(
                "Alexandra-Louise van de Zilverbergh",
                "Meisje",
                "Huidige groep met een uitzonderlijk lange naam",
                None,
            ),
            StudentEntry(
                "Bartholomeus-Jan van het Maanlicht",
                "Jongen",
                "Huidige groep met een uitzonderlijk lange naam",
                None,
            ),
        ],
        all_to_groups=["doelgroepmeteenuitzonderlijklangenaam", "anderelangdoelgroep"],
    )
    pd.DataFrame(
        {"Jongens": [1, 1], "Meisjes": [0, 0]},
        index=pd.Index(
            [
                "Doelgroep met een uitzonderlijk lange naam",
                "Andere groep in deze indeling met een uitzonderlijk lange naam",
            ],
            name="Groepen",
        ),
    ).to_excel(proc / "groups.xlsx")


@pytest.mark.usefixtures("login")
def test_processing_has_no_horizontal_overflow_at_supported_widths(
    live_server, tmp_path, page
):
    """Long controls and names remain inside the viewport on laptop and phones."""
    proc = _make_process(
        live_server, tmp_path, page, name="layout-ready", running=False
    )
    _write_long_processing_input(proc)
    for width in (1280, 390, 320):
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(f"{live_server}/processing")
        assert _has_horizontal_overflow(page)

    page.set_viewport_size({"width": 640, "height": 900})
    page.goto(f"{live_server}/processing")
    page.evaluate("document.documentElement.style.zoom = '2'")
    dimensions = page.evaluate(
        """() => ({
            html_scroll: document.documentElement.scrollWidth,
            html_client: document.documentElement.clientWidth,
            body_scroll: document.body.scrollWidth,
            body_client: document.body.clientWidth,
        })"""
    )
    assert _has_horizontal_overflow(page), dimensions


@pytest.mark.usefixtures("login")
def test_processing_controls_are_keyboard_focusable_and_motion_is_reduced(
    live_server, tmp_path, page
):
    """Primary controls, disclosures, and the live spinner respect accessibility settings."""
    _make_process(live_server, tmp_path, page, name="layout-keyboard", running=False)
    page.set_viewport_size({"width": 390, "height": 900})
    page.goto(f"{live_server}/processing")

    start = page.get_by_role("button", name="Groepsindeling berekenen →")
    start.focus()
    expect(start).to_be_focused()
    summary = page.get_by_text("Geavanceerd: maximale verschillen tussen groepen")
    summary.focus()
    page.keyboard.press("Enter")
    assert page.locator("details.instructions-box").evaluate("element => element.open")

    _make_process(live_server, tmp_path, page, name="layout-motion", running=True)
    page.emulate_media(reduced_motion="reduce")
    page.route(
        "**/status",
        lambda route: route.fulfill(
            json={
                "status_studentdistribution": "running",
                "steps": {
                    "floor": "busy",
                    "balance": "pending",
                    "satisfaction": "pending",
                },
            }
        ),
    )
    page.goto(f"{live_server}/processing")
    expect(page.locator(".loading-spinner")).to_be_visible()
    assert page.locator(".loading-spinner").evaluate(
        "element => getComputedStyle(element).animationName"
    ) in ("none", "")
