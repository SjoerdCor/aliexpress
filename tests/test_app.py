"""Tests for app-level concerns: secret key guard, upload size limit, home route, session guard."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

from io import BytesIO
from types import SimpleNamespace

import app as flask_module
from app import app as flask_app
from tests.helpers import flashes, setup_process


class TestSecretKeyGuard:
    """The startup guard refuses to run without a SECRET_KEY."""

    def test_missing_secret_key_raises(self):
        """An empty SECRET_KEY must raise at startup, not silently run unsigned."""
        import pytest  # pylint: disable=import-outside-toplevel

        with pytest.raises(RuntimeError):
            flask_module.ensure_secret_key(SimpleNamespace(config={}))

    def test_present_secret_key_does_not_raise(self):
        """A configured SECRET_KEY passes the guard without error."""
        flask_module.ensure_secret_key(SimpleNamespace(config={"SECRET_KEY": "x"}))


class TestUploadSizeLimit:
    """Uploads exceeding MAX_CONTENT_LENGTH get a friendly 413 redirect, not a crash."""

    def test_limit_is_configured(self):
        """A content-length limit must be set so uploads cannot exhaust memory/disk."""
        assert flask_app.config["MAX_CONTENT_LENGTH"]

    def test_oversized_upload_redirects_with_error_flash(
        self, client, tmp_path, monkeypatch
    ):
        """A body larger than the limit redirects back and flashes a Dutch error."""
        setup_process(client, tmp_path)
        monkeypatch.setitem(client.application.config, "MAX_CONTENT_LENGTH", 50)
        response = client.post(
            "/upload_edexml",
            data={"edexml": (BytesIO(b"x" * 5000), "edex.xml"), "jaargroep": "4"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 302
        assert any(cat == "error" and "te groot" in msg for cat, msg in flashes(client))


class TestSimpleRenders:
    """Routes that need no state and simply render a template."""

    def test_home_returns_200(self, client):
        """GET / renders the home page."""
        assert client.get("/").status_code == 200

    def test_done_returns_200(self, client):
        """GET /done renders the done page."""
        assert client.get("/done").status_code == 200

    def test_upload_edexml_get_returns_200(self, client):
        """GET /upload_edexml renders the upload page."""
        assert client.get("/upload_edexml").status_code == 200

    def test_processing_returns_200(self, client, tmp_path):
        """GET /processing renders the processing page for the active process."""
        setup_process(client, tmp_path)
        assert client.get("/processing").status_code == 200


class TestSessionGuard:
    """Routes decorated with @require_process redirect cleanly when no session is active."""

    def test_groups_to_no_session_redirects(self, client):
        """GET /groups_to without an active process flashes 'Geen actief proces' and redirects."""
        response = client.get("/groups_to")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")
        assert any(
            cat == "error" and "Geen actief proces" in msg
            for cat, msg in flashes(client)
        )

    def test_student_preferences_no_session_redirects(self, client):
        """GET /student_preferences without a session redirects to /processes."""
        response = client.get("/student_preferences")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_not_together_no_session_redirects(self, client):
        """GET /not_together without a session redirects to /processes."""
        response = client.get("/not_together")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_start_distribution_no_session_redirects(self, client):
        """GET /start_distribution without a session redirects to /processes."""
        response = client.get("/start_distribution")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")
