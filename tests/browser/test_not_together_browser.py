# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

"""Browser coverage for the accessible page-8 student-spread slice."""

import json

import pytest
from playwright.sync_api import expect

from aliexpress.data.preferences_form import StudentEntry
from aliexpress.web.extensions import db as flask_db
from aliexpress.web.models import Process
from app import app
from tests.browser.conftest import TEST_SCHOOLCODE
from tests.helpers import write_minimal_groups_xlsx, write_minimal_voorkeuren_json

LONG_NAMES = [
    "Alexandria van der Linden met een extra lange leerlingnaam",
    "Boudewijn de Vries met een extra lange leerlingnaam",
    "Charlotte van den Berg met een extra lange leerlingnaam",
]


def _open_not_together(live_server, tmp_path, page):
    """Create a settled preference input and land on page 8."""
    proc = tmp_path / TEST_SCHOOLCODE / "browsertest"
    proc.mkdir(parents=True, exist_ok=True)
    write_minimal_groups_xlsx(proc)
    write_minimal_voorkeuren_json(
        proc,
        students=[
            StudentEntry(
                student=name,
                sex="Jongen",
                origin_group="Groep 4",
                min_satisfaction=None,
            )
            for name in LONG_NAMES
        ],
        all_to_groups=["klas a", "klas b"],
    )
    (proc / "input_method.json").write_text(
        json.dumps({"method": "form"}), encoding="utf-8"
    )
    with app.app_context():
        flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name="browsertest"))
        flask_db.session.commit()
    page.goto(f"{live_server}/processes/select/browsertest")
    page.wait_for_url("**/not_together")
    return proc


def _has_no_horizontal_overflow(page):
    return page.evaluate(
        "document.scrollingElement.scrollWidth <= document.documentElement.clientWidth"
    )


def _add_students(page, names):
    """Add known students to the current, last spread using the keyboard."""
    card = page.locator(".rule-card").last
    for name in names:
        picker = card.locator(".student-combobox-input")
        picker.fill(name)
        picker.press("Enter")
        expect(card.locator(".student-chip", has_text=name)).to_be_visible()
    return card


@pytest.mark.usefixtures("login")
def test_spread_picker_keyboard_validation_and_narrow_layout(
    live_server, tmp_path, page
):
    """Known students become chips, errors stay visible, and long names wrap."""
    _open_not_together(live_server, tmp_path, page)

    for width in (1280, 390, 320):
        page.set_viewport_size({"width": width, "height": 760})
        assert _has_no_horizontal_overflow(page)

    page.get_by_role("button", name="Spreiding toevoegen").click()
    card = page.locator(".rule-card").last
    picker = card.locator(".student-combobox-input")
    picker.press("Enter")
    expect(card.locator(".student-chip").first).to_be_visible()
    _add_students(page, LONG_NAMES[1:2])

    maximum = card.locator(".rule-max-label input")
    assert maximum.input_value() == "2"
    maximum.fill("1")
    picker = card.locator(".student-combobox-input")
    picker.fill(LONG_NAMES[2])
    picker.press("Enter")
    assert maximum.input_value() == "2"
    card.get_by_role(
        "button", name=f"Verwijder {LONG_NAMES[2]} uit deze spreiding"
    ).click()
    assert maximum.input_value() == "2"

    card.get_by_role("button", name="Bevestigen").click()
    expect(card.get_by_role("button", name="Bewerken")).to_be_visible()
    expect(card.locator(".rule-add-area")).to_be_hidden()
    expect(page.get_by_role("button", name="Spreiding toevoegen")).to_be_visible()

    card.get_by_role("button", name="Bewerken").click()
    expect(card.locator(".rule-add-area")).to_be_visible()
    expect(page.get_by_role("button", name="Spreiding toevoegen")).to_be_visible()
    picker.fill("Onbekende leerling")
    picker.press("Enter")
    error = page.locator("#not-together-client-message")
    expect(error).to_be_visible()
    expect(error).to_contain_text("Kies een leerling uit de deelnemers")
    assert picker.input_value() == "Onbekende leerling"
    assert _has_no_horizontal_overflow(page)


@pytest.mark.usefixtures("login")
def test_spread_delete_renumbers_and_posts_saved_rules(live_server, tmp_path, page):
    """Deleting a middle spread keeps the remaining visible order and stored rules."""
    proc = _open_not_together(live_server, tmp_path, page)

    for names in (
        LONG_NAMES[:2],
        LONG_NAMES[1:],
        [LONG_NAMES[0], LONG_NAMES[2]],
    ):
        page.get_by_role("button", name="Spreiding toevoegen").click()
        card = _add_students(page, names)
        card.get_by_role("button", name="Bevestigen").click()

    page.locator(".rule-card").nth(1).get_by_role(
        "button", name="Spreiding verwijderen"
    ).click()
    assert page.locator(".rule-title").all_text_contents() == [
        "Spreiding 1",
        "Spreiding 2",
    ]

    page.get_by_role("button", name="Verder →").click()
    page.wait_for_url("**/processing")
    saved = json.loads((proc / "not_together.json").read_text(encoding="utf-8"))
    assert [set(rule["group"]) for rule in saved] == [
        set(LONG_NAMES[:2]),
        {LONG_NAMES[0], LONG_NAMES[2]},
    ]


@pytest.mark.usefixtures("login")
def test_empty_spread_is_not_silently_removed_on_continue(live_server, tmp_path, page):
    """Continuing with an empty spread gives repair advice and keeps the card."""
    _open_not_together(live_server, tmp_path, page)
    page.get_by_role("button", name="Spreiding toevoegen").click()
    page.get_by_role("button", name="Verder →").click()

    expect(page.locator("#not-together-client-message")).to_contain_text(
        "Voeg minimaal twee leerlingen toe"
    )
    assert page.locator(".rule-card").count() == 1
    assert page.url.endswith("/not_together")
