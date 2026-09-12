# pylint: disable=redefined-outer-name,duplicate-code  # standard pytest fixture pattern and shared browser fixture setup

"""Browser tests for the roster page ("Leerlingen controleren"): visible roster behavior —
confirming a new student, editing/removing it, and continuing the wizard.
"""

import json
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from aliexpress.web.extensions import db as flask_db
from aliexpress.web.models import Process
from app import app
from tests.browser.conftest import TEST_SCHOOLCODE
from tests.browser.wizard_route_helpers import assert_wizard_page

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
    # Candidate determination supplies the catch-all option for the roster route.
    (proc / "relevant_students_and_groups.json").write_text(
        json.dumps(
            {
                "candidates": CANDIDATES,
                "groups_from": ["Groep 3", "Anders"],
                "groups_to": {"Klas A": [], "Klas B": []},
            }
        ),
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


def _create_forward_process(live_server, page, name):
    """Create a normal Doorzetten process and open its Schoolinformatie step."""
    page.goto(f"{live_server}/processes?new=1")
    page.fill("#processName", name)
    page.click("#processForm button[type=submit]")
    page.wait_for_url(f"{live_server}/upload_edexml")


def _build_forward_edexml() -> bytes:
    """Build a small EDEXML file with one source and two next-year groups."""
    root = ET.Element("EDEX")
    groups = (
        ("SOURCE", "Groep 5A", "5"),
        ("TARGET_A", "Groep 6A", "6"),
        ("TARGET_B", "Groep 6B", "6"),
    )
    groups_element = ET.SubElement(root, "groepen")
    for key, name, year in groups:
        group_element = ET.SubElement(groups_element, "groep", key=key)
        ET.SubElement(group_element, "naam").text = name
        ET.SubElement(group_element, "jaargroep").text = year

    students_element = ET.SubElement(root, "leerlingen")
    for key, first_name, last_name, gender in (
        ("s1", "Anna", "Bos", "2"),
        ("s2", "Bram", "Dijk", "1"),
        ("t1", "Cato", "Eik", "2"),
        ("t2", "Daan", "Fijn", "1"),
    ):
        student = ET.SubElement(students_element, "leerling", key=key)
        ET.SubElement(student, "roepnaam").text = first_name
        ET.SubElement(student, "achternaam").text = last_name
        ET.SubElement(student, "geslacht").text = gender
        year = "5" if key.startswith("s") else "6"
        group = (
            "SOURCE"
            if key.startswith("s")
            else "TARGET_A" if key == "t1" else "TARGET_B"
        )
        ET.SubElement(student, "jaargroep").text = year
        ET.SubElement(student, "groep", key=group)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _upload_forward_edexml(live_server, page):
    """Upload the forward fixture and land on the learner-checking step."""
    page.set_input_files(
        "input[name=edexml]",
        {
            "name": "edex.xml",
            "mimeType": "text/xml",
            "buffer": _build_forward_edexml(),
        },
    )
    page.locator("select[name=jaargroep]").select_option("5")
    page.click("button[type=submit]")
    page.wait_for_url(f"{live_server}/roster")


def _add_student(page, voornaam, achternaam, geslacht="Meisje", confirm=True):
    """Add one new-student row, fill it, and optionally confirm it."""
    page.click("button:has-text('Leerling toevoegen')")
    row = page.locator(".new-student-row").last
    row.locator("[name='new_voornaam[]']").fill(voornaam)
    row.locator("[name='new_achternaam[]']").fill(achternaam)
    if geslacht:
        row.locator("[name='new_geslacht[]']").select_option(geslacht)
    row.locator("[name='new_groep[]']").select_option("Groep 3")
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
    assert "Groep 3" in chip.inner_text()


@pytest.mark.usefixtures("login")
def test_origin_group_dropdown_has_single_anders(live_server, tmp_path, page):
    """The backend-provided dropdown includes exactly one 'Anders' option."""
    _open_roster(live_server, tmp_path, page)
    page.click("button:has-text('Leerling toevoegen')")
    row = page.locator(".new-student-row").last
    anders = row.locator("[name='new_groep[]'] option", has_text="Anders")
    assert anders.count() == 1


@pytest.mark.usefixtures("login")
def test_forward_keeps_existing_students_in_roster(live_server, tmp_path, page):
    """Continuing to the group-checking page must carry the pre-checked existing students into the
    roster (regression: the Excel download later reported 'no students')."""
    proc = _open_roster(live_server, tmp_path, page)
    page.locator("button[type=submit]").click()
    page.wait_for_url("**/groups_to")
    roster = json.loads((proc / "roster.json").read_text("utf-8"))
    assert {p["key"] for p in roster["participants"]} == {"s1", "s2"}


@pytest.mark.usefixtures("login")
def test_forward_route_steps_and_navigation_labels(live_server, page):
    """Doorzetten follows every step from Schoolinformatie through Klaar!."""
    _create_forward_process(live_server, page, "full-forward-flow")
    assert_wizard_page(page, "forward", "Schoolinformatie")
    _upload_forward_edexml(live_server, page)
    assert_wizard_page(page, "forward", "Leerlingen controleren")

    page.locator("button.next-step").click()
    page.wait_for_url("**/groups_to")
    assert_wizard_page(page, "forward", "Groepen controleren")

    page.locator("button[name=action][value=form]").click()
    page.wait_for_url("**/preferences_form")
    assert_wizard_page(page, "forward", "Voorkeuren invullen")

    page.locator("button[name=action][value=volgende]").click()
    page.wait_for_url("**/not_together")
    assert_wizard_page(page, "forward", "Leerlingen spreiden")

    page.locator("button.next-step").click()
    page.wait_for_url("**/processing")
    assert_wizard_page(page, "forward", "Groepsindeling berekenen")

    page.get_by_role("button", name="Groepsindeling berekenen →").click()
    page.wait_for_url("**/result", timeout=60000)
    assert_wizard_page(page, "forward", "Resultaat bekijken")

    page.get_by_role("link", name="Ja, ik ben tevreden!").click()
    page.wait_for_url("**/done")
    assert_wizard_page(page, "forward", "Klaar!")


@pytest.mark.usefixtures("login")
def test_confirmed_new_student_submits(live_server, tmp_path, page):
    """A confirmed new student is written to roster.json on forward."""
    proc = _open_roster(live_server, tmp_path, page)
    _add_student(page, "Emma", "Jansen")
    page.locator("button[type=submit]").click()
    page.wait_for_url("**/groups_to")
    roster = json.loads((proc / "roster.json").read_text("utf-8"))
    assert "Emma Jansen" in {
        f"{p['roepnaam']} {p['achternaam']}" for p in roster["participants"]
    }


@pytest.mark.usefixtures("login")
def test_unconfirmed_row_blocks_submit(live_server, tmp_path, page):
    """A started-but-unconfirmed row blocks forward navigation with a message."""
    _open_roster(live_server, tmp_path, page)
    _add_student(page, "Emma", "Jansen", geslacht="", confirm=False)
    page.locator("button[type=submit]").click()
    page.wait_for_timeout(300)
    assert "/roster" in page.url
    assert page.locator("#roster-client-message").inner_text() != ""


@pytest.mark.usefixtures("login")
def test_empty_new_student_row_blocks_submit(live_server, tmp_path, page):
    """Adding a blank row still requires confirmation or removal before continuing."""
    _open_roster(live_server, tmp_path, page)
    page.get_by_role("button", name="+ Leerling toevoegen").click()
    page.locator("button[type=submit]").click()
    assert "/roster" in page.url
    assert (
        "Klik op ‘Leerling aan de lijst toevoegen’ om deze leerling te bevestigen"
        in page.locator("#roster-client-message").inner_text()
    )


@pytest.mark.usefixtures("login")
def test_roster_requires_one_participant(live_server, tmp_path, page):
    """Continuing with every existing leerling unticked is rejected."""
    _open_roster(live_server, tmp_path, page)
    for checkbox in page.locator('input[name="gaat_over"]:not([hidden])').all():
        checkbox.uncheck()
    page.locator("button[type=submit]").click()
    assert (
        "Selecteer ten minste één leerling die doorgaat."
        in page.locator("#roster-client-message").inner_text()
    )


@pytest.mark.usefixtures("login")
def test_name_collision_flagged_on_confirm(live_server, tmp_path, page):
    """Confirming a new student whose name matches an existing leerling is rejected."""
    _open_roster(live_server, tmp_path, page)
    row = _add_student(page, "Anna", "Bos")  # clashes with candidate s1
    assert row.get_attribute("data-confirmed") == "0"
    assert page.locator(".roster-client-message-text").inner_text() == (
        "Er staat al een leerling met de naam ‘Anna Bos’ in de lijst."
    )


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
    page.locator("button[type=submit]").click()
    page.wait_for_url("**/groups_to")

    page.goto(f"{live_server}/roster")
    chip = page.locator(".new-student-row .ns-chip")
    assert chip.count() == 1
    assert "Emma Jansen" in chip.first.inner_text()


@pytest.mark.usefixtures("login")
def test_roster_keyboard_and_narrow_layout(live_server, tmp_path, page):
    """The roster remains operable by keyboard at narrow widths and 200% zoom."""
    _open_roster(live_server, tmp_path, page)

    for width in (320, 390):
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(f"{live_server}/roster")
        add_button = page.get_by_role("button", name="+ Leerling toevoegen")
        add_button.focus()
        page.keyboard.press("Enter")

        row = page.locator(".new-student-row").last
        row.get_by_label("Voornaam").fill("AlexandervanDriessen")
        row.get_by_label("Achternaam").fill("VanDerLangeAchternaam")
        row.get_by_label("Geslacht").focus()
        page.keyboard.press("ArrowDown")
        row.get_by_label("Huidige groep").focus()
        page.keyboard.press("ArrowDown")
        row.get_by_role("button", name="Leerling aan de lijst toevoegen").focus()
        page.keyboard.press("Enter")

        assert row.get_by_text(
            "AlexandervanDriessen VanDerLangeAchternaam"
        ).is_visible()
        assert page.evaluate(
            "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
        )

        row.get_by_role(
            "button", name="Verwijder AlexandervanDriessen VanDerLangeAchternaam"
        ).focus()
        page.keyboard.press("Enter")
        assert page.locator(".new-student-row").count() == 0

    # A 390 CSS-pixel viewport at 200% zoom exposes roughly 195 CSS pixels to the
    # layout, which is the portable way to exercise the same reflow in Playwright.
    page.set_viewport_size({"width": 195, "height": 900})
    page.goto(f"{live_server}/roster")
    assert page.evaluate(
        """() => {
      const width = document.documentElement.clientWidth;
      return [...document.querySelectorAll('*')].every(element => {
        const rect = element.getBoundingClientRect();
        return rect.left >= -0.5 && rect.right <= width + 0.5;
      });
    }"""
    )
