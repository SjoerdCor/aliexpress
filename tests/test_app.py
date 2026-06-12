"""Tests for app.py process-management routes."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

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
