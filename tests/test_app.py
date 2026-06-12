"""Tests for app.py process-management routes."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

from io import BytesIO

import pandas as pd
import pytest

import app as flask_module
from app import app as flask_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Flask test client with BASE_DIR redirected to a fresh temporary directory."""
    monkeypatch.setattr(flask_module, "BASE_DIR", str(tmp_path))
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-key"
    with flask_app.test_client() as c:
        yield c


def _flashes(client_obj):
    """Return list of (category, message) flash tuples from the current session."""
    with client_obj.session_transaction() as sess:
        return sess.get("_flashes", [])


class TestCreateProcess:
    """Tests for POST /processes/create."""

    def test_empty_name_gives_naam_is_verplicht(self, client):
        """Bug 3: empty name must yield 'Naam is verplicht', not the regex message."""
        response = client.post("/processes/create", data={"process_name": ""})
        assert response.status_code == 302
        assert _flashes(client) == [("error", "Naam is verplicht")]

    def test_invalid_chars_gives_format_error(self, client):
        """Invalid characters yield a format error."""
        response = client.post("/processes/create", data={"process_name": "bad/name!"})
        assert response.status_code == 302
        assert _flashes(client) == [
            ("error", "Alleen letters, cijfers, spaties, - en _ toegestaan")
        ]

    def test_existing_name_gives_bestaat_al(self, client, tmp_path):
        """Bug 2: creating a duplicate must yield 'Proces bestaat al', not 'bestaat niet'."""
        (tmp_path / "mijnproces").mkdir()
        response = client.post("/processes/create", data={"process_name": "mijnproces"})
        assert response.status_code == 302
        assert _flashes(client) == [("error", "Proces bestaat al")]

    def test_happy_path_creates_directory(self, client, tmp_path):
        """A valid new name creates the process directory and redirects to upload."""
        response = client.post(
            "/processes/create", data={"process_name": "nieuwproces"}
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/upload_edexml")
        assert (tmp_path / "nieuwproces").is_dir()


class TestDeleteProcess:
    """Tests for POST /processes/delete/<process_name>."""

    def test_nonexistent_name_gives_bestaat_niet(self, client):
        """Bug 2: deleting a missing process must yield 'Proces bestaat niet', not 'bestaat al'."""
        response = client.post("/processes/delete/spookproces")
        assert response.status_code == 302
        assert _flashes(client) == [("error", "Proces bestaat niet")]

    def test_invalid_chars_gives_format_error(self, client):
        """A name with a slash hits the router before validation; expect 302 or 404."""
        response = client.post("/processes/delete/bad/name")
        assert response.status_code in (302, 404)

    def test_happy_path_removes_directory(self, client, tmp_path):
        """Deleting an existing process removes the directory and redirects."""
        (tmp_path / "teproces").mkdir()
        response = client.post("/processes/delete/teproces")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")
        assert not (tmp_path / "teproces").exists()


class TestUploadErrors:
    """Tests for friendly error handling on file upload routes."""

    def _setup_process(self, client, tmp_path, process_id="testproces"):
        """Create a process directory and set session process_id."""
        proc_dir = tmp_path / process_id
        proc_dir.mkdir()
        with client.session_transaction() as sess:
            sess["process_id"] = process_id
        return proc_dir

    def test_garbage_preferences_redirects_with_error_flash(
        self, client, tmp_path, monkeypatch
    ):
        """Uploading a garbage file as preferences flashes an error and redirects back."""
        proc_dir = self._setup_process(client, tmp_path)
        monkeypatch.setattr(
            flask_module.datareader,
            "read_groups_excel",
            lambda _path: {"Klas A": None},
        )

        response = client.post(
            "/upload_preferences",
            data={"preferences": (BytesIO(b"not an excel"), "voorkeuren.xlsx")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/student_preferences")
        flashes = _flashes(client)
        assert any(cat == "error" for cat, _msg in flashes)
        assert not (proc_dir / "preferences.xlsx").exists()

    def test_wrong_column_preferences_flashes_column_message(
        self, client, tmp_path, monkeypatch
    ):
        """An Excel with wrong columns produces the column-mismatch Dutch message."""
        self._setup_process(client, tmp_path)
        monkeypatch.setattr(
            flask_module.datareader,
            "read_groups_excel",
            lambda _path: {"Klas A": None},
        )

        # VoorkeurenProcessor reads with header=None and accesses rows 0-2 to build
        # a MultiIndex; the DataFrame must have at least 3 rows so the wrong-columns
        # path (not an IndexError) is triggered.
        buf = BytesIO()
        pd.DataFrame({"VerkeerdeKolom": ["hdr1", "hdr2", "hdr3", "data"]}).to_excel(
            buf, index=False
        )
        buf.seek(0)

        response = client.post(
            "/upload_preferences",
            data={"preferences": (buf, "voorkeuren.xlsx")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/student_preferences")
        flashes = _flashes(client)
        assert any(
            cat == "error" and "verkeerde kolommen" in msg for cat, msg in flashes
        )

    def test_garbage_edexml_redirects_with_error_flash(self, client, tmp_path):
        """Uploading a garbage EDEXML file flashes an error and redirects back."""
        self._setup_process(client, tmp_path)

        response = client.post(
            "/upload_edexml",
            data={
                "edexml": (BytesIO(b"garbage"), "edex.xml"),
                "jaargroep": "4",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/upload_edexml")
        flashes = _flashes(client)
        assert any(cat == "error" for cat, _msg in flashes)
