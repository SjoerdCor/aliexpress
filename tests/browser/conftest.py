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

import app as flask_module
from aliexpress.extensions import db as flask_db
from aliexpress.models import Process, School

TEST_SCHOOLCODE = "browser-school"
TEST_PASSWORD = "browser-pass"


@pytest.fixture
def live_server(tmp_path):
    """Run the app on a random port with an isolated storage dir; yield its base URL.

    Also creates the test school so the login fixture can authenticate.
    """
    flask_module.app.config["TESTING"] = True
    flask_module.app.config["SECRET_KEY"] = "browser-test-secret"
    flask_module.app.config["STORAGE_DIR"] = str(tmp_path)
    flask_module.limiter.enabled = False

    with flask_module.app.app_context():
        flask_module.db.drop_all()
        flask_module.db.create_all()
        school = School(
            schoolcode=TEST_SCHOOLCODE,
            naam="Browser Testschool",
            password_hash=generate_password_hash(TEST_PASSWORD),
        )
        flask_module.db.session.add(school)
        flask_module.db.session.commit()

    server = make_server("127.0.0.1", 0, flask_module.app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def _do_login(page, base_url):
    """Fill in the test-school credentials and wait for the redirect to /processes."""
    page.goto(f"{base_url}/login")
    page.fill("#schoolcode", TEST_SCHOOLCODE)
    page.fill("#wachtwoord", TEST_PASSWORD)
    page.click('button[type="submit"]')
    page.wait_for_url(f"{base_url}/processes")


@pytest.fixture
def login(live_server, page):
    """Authenticate the browser session as the test school via /login."""
    _do_login(page, live_server)


@pytest.fixture
def open_groups_to(live_server, tmp_path, page):
    """Create a process with the given groups_to and open the groups page in the browser.

    Returns the process directory so a test can read back the files the POST writes.
    """
    _do_login(page, live_server)

    def _open(groups_to):
        proc = tmp_path / TEST_SCHOOLCODE / "browsertest"
        proc.mkdir(parents=True, exist_ok=True)
        (proc / "relevant_students_and_groups.json").write_text(
            json.dumps({"groups_to": groups_to}), encoding="utf-8"
        )
        with flask_module.app.app_context():
            flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name="browsertest"))
            flask_db.session.commit()
        page.goto(f"{live_server}/processes/select/browsertest")
        return proc

    return _open
