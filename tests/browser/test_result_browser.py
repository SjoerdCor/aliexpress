"""Focused browser checks for the completed result page."""

import json

import pytest
from playwright.sync_api import expect

from tests.browser.test_distribution_browser import (
    _make_process,
    _start_distribution_from_idle_panel,
)


def _assert_no_horizontal_page_scroll(page):
    """Check the page bounds instead of a page-specific CSS implementation."""
    assert page.evaluate(
        """() => Math.max(document.body.scrollWidth, document.documentElement.scrollWidth)
        <= document.documentElement.clientWidth + 1"""
    )


def _check_result_narrow_viewports(page):
    """Check the result page at mobile and zoom-equivalent viewport widths."""
    page.set_viewport_size({"width": 390, "height": 900})
    page.reload()
    page.get_by_role("heading", name="Je groepsindeling is klaar!").wait_for()
    _assert_no_horizontal_page_scroll(page)
    student_details = page.locator("details").filter(
        has_text="Tevredenheid en voorkeuren per leerling"
    )
    student_summary = student_details.locator("summary")
    student_summary.focus()
    page.keyboard.press("Enter")
    assert student_details.evaluate("element => element.open") is True
    _assert_no_horizontal_page_scroll(page)
    page.keyboard.press("Enter")
    assert student_details.evaluate("element => element.open") is False
    first_chip = page.locator(".gi-chip[tabindex='0']").first
    first_chip.click()
    expect(first_chip.locator(".gi-pop")).to_be_visible()
    _assert_no_horizontal_page_scroll(page)
    page.keyboard.press("Escape")

    page.set_viewport_size({"width": 320, "height": 900})
    page.reload()
    page.get_by_role("heading", name="Je groepsindeling is klaar!").wait_for()
    _assert_no_horizontal_page_scroll(page)

    # A 640 CSS-pixel viewport is the layout width seen at 200% zoom on a
    # 1280-pixel display.
    page.set_viewport_size({"width": 640, "height": 900})
    page.reload()
    page.get_by_role("heading", name="Je groepsindeling is klaar!").wait_for()
    _assert_no_horizontal_page_scroll(page)


def _capture_result_review_images(page):
    """Capture the review artefacts outside the repository."""
    page.set_viewport_size({"width": 1280, "height": 900})
    page.reload()
    page.get_by_role("heading", name="Je groepsindeling is klaar!").wait_for()
    page.screenshot(path="/tmp/aliexpress-result-review-laptop.png", full_page=True)
    page.set_viewport_size({"width": 390, "height": 900})
    page.reload()
    page.get_by_role("heading", name="Je groepsindeling is klaar!").wait_for()
    page.screenshot(path="/tmp/aliexpress-result-review-390.png", full_page=True)


@pytest.mark.usefixtures("login")
@pytest.mark.real_solver
def test_result_page_is_native_and_stays_inside_narrow_viewports(
    live_server, tmp_path, page
):
    """The result page exposes native analyses and keeps long details in the page."""
    proc = _make_process(
        live_server, tmp_path, page, name="result-browser", running=False
    )
    _start_distribution_from_idle_panel(live_server, page)
    page.wait_for_url("**/result", timeout=60000)

    view_path = proc / "groepsindeling_view.json"
    view = json.loads(view_path.read_text(encoding="utf-8"))
    long_name = "Alexandria van der Lange-Namen met een bijzonder lange achternaam"
    first_student = view["groups"][0]["year_sections"][0]["boys"]["students"][0]
    first_student["chip_name"] = long_name
    first_student["full_name"] = long_name
    view_path.write_text(json.dumps(view), encoding="utf-8")
    page.reload()
    page.get_by_role("heading", name="Je groepsindeling is klaar!").wait_for()

    expect(
        page.get_by_role("heading", name="Je groepsindeling is klaar!")
    ).to_be_visible()
    student_details = page.locator("details").filter(
        has_text="Tevredenheid en voorkeuren per leerling"
    )
    origin_details = page.locator("details").filter(
        has_text="Waar komen de leerlingen vandaan?"
    )
    assert student_details.evaluate("element => element.open") is False
    assert origin_details.evaluate("element => element.open") is False
    expect(page.get_by_text("Uitleg bij de groepskaarten")).to_be_visible()

    student_details.locator("summary").click()
    expect(student_details.locator("li").first).to_be_visible()
    assert student_details.locator("li").count() > 0
    origin_details.locator("summary").click()
    expect(origin_details.get_by_role("table")).to_be_visible()
    assert origin_details.locator("tfoot").count() == 0

    expect(page.get_by_role("link", name="Ja, ik ben tevreden!")).to_have_attribute(
        "href", "/done"
    )
    _check_result_narrow_viewports(page)
    _capture_result_review_images(page)


@pytest.mark.usefixtures("login")
@pytest.mark.real_solver
def test_result_adjustment_links_follow_saved_input_method_and_open_limits(
    live_server, tmp_path, page
):
    """Adjustment links reuse the saved preference route and existing processing form."""
    proc = _make_process(
        live_server, tmp_path, page, name="result-adjustments", running=False
    )
    _start_distribution_from_idle_panel(live_server, page)
    page.wait_for_url("**/result", timeout=60000)

    workbook = proc / "results.xlsx"
    assert workbook.exists()
    workbook.unlink()
    page.get_by_role("link", name="Download als Excel-bestand").click()
    page.wait_for_url("**/result")
    expect(page.get_by_text("Groepsindeling niet gevonden")).to_be_visible()

    adjustment = page.locator("details").filter(has_text="Nog niet helemaal")
    adjustment.locator("summary").click()
    expect(page.locator("a", has_text="Voorkeuren aanpassen")).to_have_attribute(
        "href", "/preferences_form"
    )
    expect(
        page.locator("a", has_text="Leerlingen spreiden aanpassen")
    ).to_have_attribute("href", "/not_together")

    (proc / "input_method.json").write_text(
        json.dumps({"method": "excel"}), encoding="utf-8"
    )
    page.reload()
    adjustment.locator("summary").click()
    expect(page.locator("a", has_text="Voorkeuren aanpassen")).to_have_attribute(
        "href", "/preferences_excel"
    )

    page.locator(
        "a", has_text="Ruimte voor verschillen tussen groepen aanpassen"
    ).first.click(no_wait_after=True)
    page.wait_for_url(f"{live_server}/processing?edit=differences")
    details = page.locator("details.instructions-box")
    expect(details).to_be_visible()
    assert details.evaluate("element => element.open") is True
