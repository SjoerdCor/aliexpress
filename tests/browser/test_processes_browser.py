"""Browser checks for the two states of the grouping-process page."""

import pytest


@pytest.mark.usefixtures("login")
def test_existing_processes_open_resume_state_and_new_state(live_server, page):
    """Existing work is the default view; the new form opens deliberately."""
    page.goto(f"{live_server}/processes?new=1", wait_until="domcontentloaded")
    page.fill("#processName", "Overgang 5 2026")
    page.check("input[name='mode'][value='redistribute']")
    page.click("#processForm button[type=submit]", no_wait_after=True)
    page.wait_for_url(f"{live_server}/upload_edexml", wait_until="domcontentloaded")

    page.goto(f"{live_server}/processes", wait_until="domcontentloaded")
    assert page.locator("h1").inner_text() == "Jouw groepsindelingen"
    assert page.locator("#processForm").count() == 0
    assert page.get_by_role("link", name="Verder met Overgang 5 2026").count() == 1
    assert (
        page.get_by_role(
            "button", name="Groepsindeling Overgang 5 2026 verwijderen"
        ).count()
        == 1
    )
    assert page.locator("[data-tooltip='Verwijderen']").count() == 1

    page.get_by_role("link", name="Begin een nieuwe groepsindeling").click()
    assert page.locator("h1").inner_text() == "Nieuwe groepsindeling maken"
    assert page.locator("#processForm").count() == 1
    assert page.locator("input[name='mode'][value='forward']").is_checked()


@pytest.mark.usefixtures("login")
def test_new_process_validation_keeps_name_and_mode(live_server, page):
    """A validation error returns to the new state with the entered choices intact."""
    page.goto(f"{live_server}/processes?new=1", wait_until="domcontentloaded")
    page.fill("#processName", "Ongeldige/naam")
    page.check("input[name='mode'][value='redistribute_and_forward']")
    assert page.locator("#process-name-rules").is_visible()
    page.click("#processForm button[type=submit]", no_wait_after=True)

    page.wait_for_url("**/processes?new=1", wait_until="domcontentloaded")
    assert page.locator("#processName").input_value() == "Ongeldige/naam"
    assert page.locator(
        "input[name='mode'][value='redistribute_and_forward']"
    ).is_checked()


@pytest.mark.usefixtures("login")
def test_processes_mobile_long_name_and_keyboard_focus(live_server, page):
    """The compact overview remains usable at 390px and its actions stay keyboard-visible."""
    page.set_viewport_size({"width": 390, "height": 844})
    long_name = "Overgang " + "x" * 55

    page.goto(f"{live_server}/processes?new=1", wait_until="domcontentloaded")
    page.fill("#processName", long_name)
    page.click("#processForm button[type=submit]", no_wait_after=True)
    page.wait_for_url(f"{live_server}/upload_edexml", wait_until="domcontentloaded")

    page.goto(f"{live_server}/processes", wait_until="domcontentloaded")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert page.get_by_role("link", name=f"Verder met {long_name}").is_visible()
    delete_button = page.get_by_role(
        "button", name=f"Groepsindeling {long_name} verwijderen"
    )
    assert delete_button.is_visible()

    page.goto(f"{live_server}/processes?new=1", wait_until="domcontentloaded")
    back_link = page.get_by_role("link", name="← Jouw groepsindelingen")
    back_link.focus()
    page.keyboard.press("Enter")
    page.wait_for_url(f"{live_server}/processes?clear=1", wait_until="domcontentloaded")

    delete_button = page.get_by_role(
        "button", name=f"Groepsindeling {long_name} verwijderen"
    )
    delete_button.focus()
    assert delete_button.evaluate("element => document.activeElement === element")
    # Let the 100ms opacity transition settle before inspecting the pseudo-element.
    page.wait_for_timeout(150)
    tooltip_style = delete_button.evaluate(
        "element => { const style = getComputedStyle(element, '::after'); "
        "return {opacity: style.opacity, content: style.content}; }"
    )
    assert float(tooltip_style["opacity"]) > 0, tooltip_style
