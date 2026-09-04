# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

"""Fixtures for the Playwright browser tests.

These drive the real Flask app in a background thread so the page JavaScript (group
enable/disable, adding groups, the preferences forward button) is exercised end to end.
They are kept out of the quick test loop; run them explicitly with

    uv run pytest tests/browser
"""

import json
import threading

import pytest
from werkzeug.security import generate_password_hash
from werkzeug.serving import make_server

from aliexpress.web.extensions import db as flask_db
from aliexpress.web.extensions import limiter
from aliexpress.web.models import Process, School
from app import app

TEST_SCHOOLCODE = "browser-school"
TEST_PASSWORD = "browser-pass"


@pytest.fixture(scope="session")
def live_server():
    """Run one Flask server per pytest worker and yield its base URL."""
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "browser-test-secret"
    limiter.enabled = False

    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture(scope="session")
def browser_test_password_hash():
    """Create one cheap, valid password hash per pytest worker."""
    return generate_password_hash(TEST_PASSWORD, method="pbkdf2:sha256:1")


@pytest.fixture(autouse=True)
def browser_test_state(tmp_path, live_server, browser_test_password_hash):
    """Reset database and storage for each browser test.

    The server is deliberately session-scoped, while all mutable application state
    remains function-scoped. This keeps tests isolated without restarting Flask for
    every test.
    """
    del live_server  # dependency makes fixture order explicit
    app.config["STORAGE_DIR"] = str(tmp_path)

    with app.app_context():
        flask_db.session.remove()
        flask_db.drop_all()
        flask_db.create_all()
        school = School(
            schoolcode=TEST_SCHOOLCODE,
            naam="Browser Testschool",
            password_hash=browser_test_password_hash,
        )
        flask_db.session.add(school)
        flask_db.session.commit()

    try:
        yield
    finally:
        with app.app_context():
            flask_db.session.remove()


def _do_login(page, base_url):
    """Fill in the test-school credentials and wait for the redirect to /processes."""
    page.goto(f"{base_url}/login")
    page.fill("#schoolcode", TEST_SCHOOLCODE)
    page.fill("#wachtwoord", TEST_PASSWORD)
    page.click('button[type="submit"]')
    page.wait_for_url(f"{base_url}/processes")


@pytest.fixture
def login(browser_test_state, live_server, page):
    """Authenticate the browser session as the test school via /login."""
    del browser_test_state  # dependency makes fixture order explicit
    _do_login(page, live_server)


@pytest.fixture
def open_groups_to(browser_test_state, live_server, tmp_path, page):
    """Create a process with the given groups_to and open the groups page in the browser.

    Returns the process directory so a test can read back the files the POST writes.
    """
    del browser_test_state  # dependency makes fixture order explicit
    _do_login(page, live_server)

    def _open(groups_to):
        proc = tmp_path / TEST_SCHOOLCODE / "browsertest"
        proc.mkdir(parents=True, exist_ok=True)
        (proc / "relevant_students_and_groups.json").write_text(
            json.dumps({"groups_to": groups_to}), encoding="utf-8"
        )
        # "Groepen naartoe" sits after the roster step (ADR 0006) and continues to the
        # preferences page, which needs a settled roster; provide an empty one so the
        # forward navigation lands cleanly instead of bouncing back to /roster.
        (proc / "roster.json").write_text(
            json.dumps({"participants": []}), encoding="utf-8"
        )
        with app.app_context():
            flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name="browsertest"))
            flask_db.session.commit()
        page.goto(f"{live_server}/processes/select/browsertest")
        page.goto(f"{live_server}/groups_to")
        return proc

    return _open
