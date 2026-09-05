"""Browser coverage for the client-side portable identifier checks."""


def _validity(page, selector):
    """Return the browser validity state and message for an input."""
    return page.locator(selector).evaluate(
        "element => ({valid: element.validity.valid, message: element.validationMessage})"
    )


def test_process_name_is_validated_in_browser_without_changing_case(
    login, live_server, page
):
    """Client validation rejects reserved names and keeps a valid capitalized name."""
    del login
    page.goto(f"{live_server}/processes")
    field = page.locator("#processName")

    field.fill("CON")
    invalid = _validity(page, "#processName")
    assert invalid["valid"] is False
    assert "gereserveerde" in invalid["message"]

    field.fill("Klas")
    assert field.input_value() == "Klas"
    assert _validity(page, "#processName")["valid"] is True


def test_process_name_client_validation_accepts_unicode_and_rejects_overlong_values(
    login, live_server, page
):
    """NFC-equivalent letters are accepted while the 64-character limit is enforced."""
    del login
    page.goto(f"{live_server}/processes")
    field = page.locator("#processName")

    decomposed = "e\u0301cole"
    field.fill(decomposed)
    assert field.input_value() == decomposed
    assert _validity(page, "#processName")["valid"] is True

    field.evaluate(
        """element => {
            element.value = "a".repeat(65);
            element.dispatchEvent(new Event("input", {bubbles: true}));
        }"""
    )
    invalid = _validity(page, "#processName")
    assert invalid["valid"] is False
    assert "maximaal 64" in invalid["message"]


def test_login_schoolcode_uses_the_same_client_validation(live_server, page):
    """The browser also rejects an unsafe schoolcode before making a login request."""
    page.goto(f"{live_server}/login")
    field = page.locator("#schoolcode")
    field.fill(r"C:\school")

    invalid = _validity(page, "#schoolcode")
    assert invalid["valid"] is False
    assert "letters" in invalid["message"]
