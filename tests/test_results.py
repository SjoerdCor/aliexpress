"""Tests for routes/results.py (results blueprint)."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import aliexpress.web.routes.results as results_module
from aliexpress.web.extensions import db
from aliexpress.web.models import Process
from app import app as flask_app

SCHOOL_ID = "test-school"


def _setup_process(client, tmp_path, process_id="testproces"):
    proc_dir = tmp_path / SCHOOL_ID / process_id
    proc_dir.mkdir(parents=True, exist_ok=True)
    with flask_app.app_context():
        proc = Process(school_id=SCHOOL_ID, name=process_id)
        db.session.add(proc)
        db.session.commit()
    with client.session_transaction() as sess:
        sess["process_id"] = process_id
    return proc_dir


class TestDownloadPreferences:
    """Tests for GET /download_preferences (process-scoped)."""

    def test_missing_file_redirects_via_404_handler(
        self, client, tmp_path, monkeypatch
    ):
        """When the stored preferences file is absent the 404 handler redirects to /processes."""
        _setup_process(client, tmp_path)
        monkeypatch.setattr(
            results_module, "get_file_path", lambda *_: "/nonexistent.xlsx"
        )
        response = client.get("/download_preferences")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_existing_file_sends_attachment(self, client, tmp_path):
        """When preferences.xlsx exists it is sent as a download attachment."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "preferences.xlsx").write_bytes(b"dummy preferences")
        response = client.get("/download_preferences")
        assert response.status_code == 200
        assert "attachment" in response.headers.get("Content-Disposition", "")
