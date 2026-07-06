# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

"""Browser tests for the herindelen (redistribute) flow: mode selection through the result
page. Complements test_roster_browser.py (doorzetten) and test_distribution_browser.py
(forward-mode end-to-end), which do not exercise this mode at all.
"""

import json
import xml.etree.ElementTree as ET

import pytest

from tests.browser.conftest import TEST_SCHOOLCODE
from tests.browser.test_roster_browser import _open_roster

# Three combi groups (jaargroep 6 and 7 mixed, as real combination classes are), twelve
# students total. A group's own <jaargroep> is dropped by EdexReader.get_full_df (the
# student's own <jaargroep> wins), so one group key can freely mix cohorts.
_HERINDELEN_GROUPS = [
    {"key": "GRP_A", "naam": "6-7 Alpacas (Juf Nora)"},
    {"key": "GRP_B", "naam": "6-7 Beren (Meester Tim)"},
    {"key": "GRP_C", "naam": "6-7 Ceders (Juf Mia)"},
]


def _student(fields):
    """Unpack a compact (key, roepnaam, achternaam, geslacht, jaargroep, groep) tuple."""
    key, roepnaam, achternaam, geslacht, jaargroep, groep = fields
    return {
        "key": key,
        "roepnaam": roepnaam,
        "achternaam": achternaam,
        "geslacht": geslacht,
        "jaargroep": jaargroep,
        "groep": groep,
    }


_HERINDELEN_STUDENTS = [
    _student(("h01", "Anna", "Berg", "2", "6", "GRP_A")),
    _student(("h02", "Bram", "Dijk", "1", "6", "GRP_A")),
    _student(("h03", "Clara", "Groot", "2", "7", "GRP_A")),
    _student(("h04", "Daan", "Hoek", "1", "7", "GRP_A")),
    _student(("h05", "Emma", "Jansen", "2", "6", "GRP_B")),
    _student(("h06", "Finn", "Kuiper", "1", "6", "GRP_B")),
    _student(("h07", "Gina", "Laan", "2", "7", "GRP_B")),
    _student(("h08", "Hugo", "Mulder", "1", "7", "GRP_B")),
    _student(("h09", "Iris", "Naald", "2", "6", "GRP_C")),
    _student(("h10", "Jesse", "Otter", "1", "6", "GRP_C")),
    _student(("h11", "Kim", "Prins", "2", "7", "GRP_C")),
    _student(("h12", "Lars", "Roos", "1", "7", "GRP_C")),
]


def _build_herindelen_edexml() -> bytes:
    """Build a minimal EDEXML blob that EdexReader can parse: 3 combi groups, 12 students."""
    root = ET.Element("EDEX")

    groepen_el = ET.SubElement(root, "groepen")
    for g in _HERINDELEN_GROUPS:
        groep_el = ET.SubElement(groepen_el, "groep", key=g["key"])
        ET.SubElement(groep_el, "naam").text = g["naam"]
        ET.SubElement(groep_el, "jaargroep").text = "6"

    leerlingen_el = ET.SubElement(root, "leerlingen")
    for s in _HERINDELEN_STUDENTS:
        ll = ET.SubElement(leerlingen_el, "leerling", key=s["key"])
        ET.SubElement(ll, "roepnaam").text = s["roepnaam"]
        ET.SubElement(ll, "achternaam").text = s["achternaam"]
        ET.SubElement(ll, "geslacht").text = s["geslacht"]
        ET.SubElement(ll, "jaargroep").text = s["jaargroep"]
        ET.SubElement(ll, "groep", key=s["groep"])

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _create_redistribute_process(live_server, page, name):
    """Drive /processes to create a process with the redistribute mode radio checked."""
    page.goto(f"{live_server}/processes")
    page.fill("#processName", name)
    page.check("input[name='mode'][value='redistribute']")
    page.click("#processForm button[type=submit]")
    page.wait_for_url(f"{live_server}/upload_edexml")


def _upload_herindelen_edexml(live_server, page):
    """Upload the dummy EDEXML on /upload_edexml and land on /select_groups."""
    page.set_input_files(
        "input[name=edexml]",
        {
            "name": "edex.xml",
            "mimeType": "text/xml",
            "buffer": _build_herindelen_edexml(),
        },
    )
    page.click("button[type=submit]")
    page.wait_for_url(f"{live_server}/select_groups")


def _select_groups(live_server, page, group_names):
    """Check the given groups (by display name) on /select_groups, submit, land on /roster."""
    for name in group_names:
        page.locator(f'input[name=groups][value="{name}"]').check()
    page.click("button[type=submit]")
    page.wait_for_url(f"{live_server}/roster")


def _reach_roster(live_server, page, name):
    """Create a redistribute process, upload the dummy EDEXML, select every group, land on
    /roster with the 3 combi groups and their 12 students as candidates."""
    _create_redistribute_process(live_server, page, name)
    _upload_herindelen_edexml(live_server, page)
    _select_groups(live_server, page, [g["naam"] for g in _HERINDELEN_GROUPS])


@pytest.mark.usefixtures("login")
def test_create_process_redistribute_mode(live_server, tmp_path, page):
    """Choosing "Herindelen binnen dezelfde groepen" writes mode.json and skips the
    jaargroep field on the upload page."""
    _create_redistribute_process(live_server, page, "modus-test")

    proc = tmp_path / TEST_SCHOOLCODE / "modus-test"
    assert json.loads((proc / "mode.json").read_text("utf-8")) == {
        "mode": "redistribute"
    }
    assert page.locator("select[name=jaargroep]").count() == 0
    assert (
        "Ga door naar groepskeuze" in page.locator("button[type=submit]").inner_text()
    )


@pytest.mark.usefixtures("login")
def test_upload_edexml_redistribute_shows_groups_on_select_groups(live_server, page):
    """Uploading EDEXML in redistribute mode redirects to /select_groups, listing the
    groups found in the file as checkboxes."""
    _create_redistribute_process(live_server, page, "upload-test")
    _upload_herindelen_edexml(live_server, page)

    checkboxes = page.locator("input[type=checkbox][name=groups]")
    assert checkboxes.count() == 3
    for group in _HERINDELEN_GROUPS:
        assert (
            page.locator("label", has_text=group["naam"]).count() == 1
        ), f"{group['naam']} not shown as a checkbox label"


@pytest.mark.usefixtures("login")
def test_select_groups_requires_at_least_two(live_server, page):
    """Selecting fewer than two groups flashes a warning and stays on the page; two or
    more groups proceeds to /roster."""
    _create_redistribute_process(live_server, page, "select-test")
    _upload_herindelen_edexml(live_server, page)

    first_group = _HERINDELEN_GROUPS[0]["naam"]
    second_group = _HERINDELEN_GROUPS[1]["naam"]
    page.locator(f'input[name=groups][value="{first_group}"]').check()
    page.click("button[type=submit]")
    assert (
        "Selecteer minimaal twee groepen" in page.locator(".flash-message").inner_text()
    )
    assert page.url.endswith("/select_groups")

    # The failed POST redirected to a fresh GET, so no checkbox is still checked.
    page.locator(f'input[name=groups][value="{first_group}"]').check()
    page.locator(f'input[name=groups][value="{second_group}"]').check()
    page.click("button[type=submit]")
    page.wait_for_url(f"{live_server}/roster")


@pytest.mark.usefixtures("login")
def test_roster_new_student_jaargroep_dropdown(live_server, page):
    """A hand-added student in redistribute mode gets a jaargroep dropdown, populated with
    the jaargroepen of the selected groups; confirming without a jaargroep is rejected.
    """
    _reach_roster(live_server, page, "roster-jaargroep-test")

    page.click("button:has-text('Leerling toevoegen')")
    row = page.locator(".new-student-row").last
    jaargroep_select = row.locator("[name='new_jaargroep[]']")
    option_labels = jaargroep_select.locator("option").all_inner_texts()
    assert option_labels == ["— jaargroep —", "Jaargroep 6", "Jaargroep 7"]

    # Error path: confirming without a jaargroep is rejected, the row stays unconfirmed.
    row.locator("[name='new_voornaam[]']").fill("Mila")
    row.locator("[name='new_achternaam[]']").fill("Visser")
    row.locator("[name='new_geslacht[]']").select_option("Meisje")
    row.locator("button.ns-confirm").click()
    assert row.get_attribute("data-confirmed") == "0"
    assert "jaargroep" in row.locator(".ns-error").inner_text().lower()

    # Happy path: picking a jaargroep and confirming turns the row into a chip.
    jaargroep_select.select_option("6")
    row.locator("button.ns-confirm").click()
    assert row.get_attribute("data-confirmed") == "1"
    assert "jaargroep 6" in row.locator(".ns-chip").inner_text().lower()


@pytest.mark.usefixtures("login")
def test_roster_no_jaargroep_dropdown_in_forward_mode(live_server, tmp_path, page):
    """Regression guard for the doorzetten path: the jaargroep dropdown is redistribute-only
    (complements test_roster_browser.py, which never asserts its absence)."""
    _open_roster(live_server, tmp_path, page)
    page.click("button:has-text('Leerling toevoegen')")
    row = page.locator(".new-student-row").last
    assert row.locator("[name='new_jaargroep[]']").count() == 0
