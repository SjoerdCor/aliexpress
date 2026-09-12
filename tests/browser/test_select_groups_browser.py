# pylint: disable=redefined-outer-name,duplicate-code  # standard fixture and shared upload setup

"""Focused browser acceptance tests for the group-selection page."""

import xml.etree.ElementTree as ET

import pytest

from tests.browser.test_herindelen_browser import (
    _build_herindelen_edexml,
    _create_redistribute_process,
)

_LONG_GROUP_NAMES = [
    "6-7 Alpacas met een extra lange groepsnaam voor de ochtendgroep",
    "6-7 Beren met een extra lange groepsnaam voor de middaggroep",
    "6-7 Ceders met een extra lange groepsnaam voor de combinatiegroep",
]


def _build_long_name_edexml() -> bytes:
    """Use the shared valid fixture while making each group name deliberately long."""
    root = ET.fromstring(_build_herindelen_edexml())
    for group, name in zip(root.findall("./groepen/groep"), _LONG_GROUP_NAMES):
        group.find("naam").text = name
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _upload_long_name_edexml(live_server, page):
    """Upload the long-name fixture and wait for group selection."""
    page.set_input_files(
        "input[name=edexml]",
        {
            "name": "edex.xml",
            "mimeType": "text/xml",
            "buffer": _build_long_name_edexml(),
        },
    )
    page.click("button[type=submit]")
    page.wait_for_url(f"{live_server}/select_groups")


def _has_no_horizontal_overflow(page):
    """Return whether the document fits the current viewport."""
    return page.evaluate(
        "document.scrollingElement.scrollWidth <= document.documentElement.clientWidth"
    )


@pytest.mark.usefixtures("login")
def test_select_groups_keyboard_long_names_and_zoom_fit(live_server, page):
    """The checklist stays readable and keyboard-operable on a narrow zoomed viewport."""
    _create_redistribute_process(live_server, page, "select-groups-accessibility-test")
    _upload_long_name_edexml(live_server, page)

    page.set_viewport_size({"width": 320, "height": 720})
    for name in _LONG_GROUP_NAMES:
        label = page.locator("label", has_text=name)
        assert label.count() == 1
        assert name in label.inner_text()
    assert _has_no_horizontal_overflow(page)

    first_checkbox = page.locator("input[name=groups]").first
    second_checkbox = page.locator("input[name=groups]").nth(1)
    first_checkbox.focus()
    page.keyboard.press("Space")
    assert first_checkbox.is_checked()
    page.keyboard.press("Tab")
    assert second_checkbox.evaluate("element => element === document.activeElement")

    page.set_viewport_size({"width": 390, "height": 720})
    assert _has_no_horizontal_overflow(page)
    page.evaluate("document.documentElement.style.zoom = '2'")
    assert _has_no_horizontal_overflow(page)
