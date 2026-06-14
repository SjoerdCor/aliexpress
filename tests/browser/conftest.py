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
from werkzeug.serving import make_server

import app as flask_module


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    """Run the app on a random port with an isolated storage dir; yield its base URL."""
    monkeypatch.setattr(flask_module, "BASE_DIR", str(tmp_path))
    flask_module.app.config["TESTING"] = True
    flask_module.app.config["SECRET_KEY"] = "browser-test-secret"

    # The database lives in a throwaway file (see tests/conftest.py); reset its tables.
    with flask_module.app.app_context():
        flask_module.db.drop_all()
        flask_module.db.create_all()

    server = make_server("127.0.0.1", 0, flask_module.app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture
def open_groups_to(live_server, tmp_path, page):
    """Create a process with the given groups_to and open the groups page in the browser.

    Returns the process directory so a test can read back the files the POST writes.
    """

    def _open(groups_to):
        proc = tmp_path / "browsertest"
        proc.mkdir(exist_ok=True)
        (proc / "relevant_students_and_groups.json").write_text(
            json.dumps({"groups_to": groups_to}), encoding="utf-8"
        )
        page.goto(f"{live_server}/processes/select/browsertest")
        return proc

    return _open
