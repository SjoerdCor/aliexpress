"""Playwright browser test for the login flow.

Covers: login page accessible without a session, wrong credentials show Dutch error,
correct credentials redirect to /processes, and after logout the wall is back up.
"""

import pytest

from tests.browser.conftest import TEST_PASSWORD, TEST_SCHOOLCODE


def test_login_page_is_accessible_without_session(live_server, page):
    """GET /login is public and renders the Dutch login form."""
    page.goto(f"{live_server}/login")
    assert page.locator("label", has_text="Schoolcode").is_visible()
    assert page.locator("label", has_text="Wachtwoord").is_visible()


def test_wrong_password_shows_dutch_error(live_server, page):
    """A wrong password re-renders the login page with the Dutch error."""
    page.goto(f"{live_server}/login")
    page.fill("#schoolcode", TEST_SCHOOLCODE)
    page.fill("#wachtwoord", "fout")
    page.click('button[type="submit"]')
    page.wait_for_url(f"{live_server}/login")
    assert "Ongeldige schoolcode" in page.locator(".flash-message").inner_text()


def test_correct_login_redirects_to_processes(live_server, page):
    """Correct credentials redirect to /processes."""
    page.goto(f"{live_server}/login")
    page.fill("#schoolcode", TEST_SCHOOLCODE)
    page.fill("#wachtwoord", TEST_PASSWORD)
    page.click('button[type="submit"]')
    page.wait_for_url(f"{live_server}/processes")
    assert page.url.endswith("/processes")


@pytest.mark.usefixtures("login")
def test_after_logout_protected_route_redirects_to_login(live_server, page):
    """After logging out, visiting a data route sends the browser back to /login."""
    page.goto(f"{live_server}/logout")
    page.wait_for_url(f"{live_server}/login")  # no ?next= after intentional logout
    page.goto(f"{live_server}/processes")
    page.wait_for_url(f"{live_server}/login**")  # /login?next=... after auth redirect
    assert "/login" in page.url
