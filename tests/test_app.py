"""Tests for app.py Flask routes."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
from collections import defaultdict
from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd
import pytest

import app as flask_module
from aliexpress.errors import ValidationError
from app import app as flask_app
from app import readableerror_to_validation_message, to_validation_message


def _default_status():
    return {
        "status_studentdistribution": "pending",
        "status_sociogram": "pending",
        "logs": [],
    }


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Flask test client with BASE_DIR, temp_storage and status_dct isolated per test."""
    monkeypatch.setattr(flask_module, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(flask_module, "temp_storage", {})
    monkeypatch.setattr(flask_module, "status_dct", defaultdict(_default_status))
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-key"
    with flask_app.test_client() as c:
        yield c


def _flashes(client_obj):
    """Return list of (category, message) flash tuples from the current session."""
    with client_obj.session_transaction() as sess:
        return sess.get("_flashes", [])


def _setup_process(client, tmp_path, process_id="testproces"):
    """Create a process directory and set session process_id."""
    proc_dir = tmp_path / process_id
    proc_dir.mkdir(exist_ok=True)
    with client.session_transaction() as sess:
        sess["process_id"] = process_id
    return proc_dir


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
        return _setup_process(client, tmp_path, process_id)

    def test_garbage_preferences_redirects_with_error_flash(
        self, client, tmp_path, monkeypatch
    ):
        """Uploading a garbage file as preferences flashes an error and redirects back."""
        proc_dir = self._setup_process(client, tmp_path)
        monkeypatch.setattr(
            flask_module.datareader,
            "read_groups_excel",
            lambda _path: ({"Klas A": None}, {"Klas A": "Klas A"}),
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
            lambda _path: ({"Klas A": None}, {"Klas A": "Klas A"}),
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

    def test_processing_returns_200(self, client):
        """GET /processing/<task_id> renders the processing page for any task_id."""
        assert client.get("/processing/some-task-id").status_code == 200


class TestProcessesList:
    """Tests for GET /processes."""

    def test_empty_dir_returns_200(self, client):
        """An empty BASE_DIR produces an empty process list without errors."""
        assert client.get("/processes").status_code == 200

    def test_existing_process_is_shown(self, client, tmp_path):
        """A process directory that exists appears in the processes list."""
        (tmp_path / "mijnklas").mkdir()
        response = client.get("/processes")
        assert response.status_code == 200
        assert b"mijnklas" in response.data


class TestSelectProcess:
    """Tests for GET /processes/select/<process_id>."""

    def test_unknown_process_gives_404(self, client):
        """Selecting a process that does not exist returns 404."""
        assert client.get("/processes/select/bestaat_niet").status_code == 404

    def test_empty_process_redirects_to_upload_edexml(self, client, tmp_path):
        """A process with no files starts at the first step: upload EDEXML."""
        (tmp_path / "leegproces").mkdir()
        response = client.get("/processes/select/leegproces")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/upload_edexml")

    def test_process_with_json_redirects_to_groups_to(self, client, tmp_path):
        """A process that has the candidates JSON continues at groups_to."""
        proc_dir = tmp_path / "procesmetjson"
        proc_dir.mkdir()
        (proc_dir / "relevant_students_and_groups.json").write_text(
            "{}", encoding="utf-8"
        )
        response = client.get("/processes/select/procesmetjson")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/groups_to")

    def test_process_with_groups_xlsx_redirects_to_student_preferences(
        self, client, tmp_path
    ):
        """A process that has groups.xlsx but no preferences continues at student_preferences."""
        proc_dir = tmp_path / "procesmetgroepen"
        proc_dir.mkdir()
        (proc_dir / "groups.xlsx").write_bytes(b"dummy")
        response = client.get("/processes/select/procesmetgroepen")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/student_preferences")

    def test_process_with_preferences_xlsx_redirects_to_not_together(
        self, client, tmp_path
    ):
        """A process that has preferences.xlsx continues at not_together."""
        proc_dir = tmp_path / "procesmetpref"
        proc_dir.mkdir()
        (proc_dir / "preferences.xlsx").write_bytes(b"dummy")
        response = client.get("/processes/select/procesmetpref")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/not_together")


class TestSessionGuard:
    """Routes decorated with @require_process redirect cleanly when no session is active."""

    def test_groups_to_no_session_redirects(self, client):
        """GET /groups_to without an active process flashes 'Geen actief proces' and redirects."""
        response = client.get("/groups_to")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")
        assert any(
            cat == "error" and "Geen actief proces" in msg
            for cat, msg in _flashes(client)
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


class TestGroupsToPage:
    """Tests for GET/POST /groups_to."""

    def test_get_renders_groups_from_json(self, client, tmp_path):
        """GET /groups_to reads the candidates JSON and renders group names in the page."""
        proc_dir = _setup_process(client, tmp_path)
        # groups_to is a dict {groupname: [students]}; the template calls .items()
        (proc_dir / "relevant_students_and_groups.json").write_text(
            json.dumps({"groups_to": {"Klas A": [], "Klas B": []}}), encoding="utf-8"
        )
        response = client.get("/groups_to")
        assert response.status_code == 200
        assert b"Klas A" in response.data

    def test_post_too_few_groups_flashes_error(self, client, tmp_path):
        """POST /groups_to with fewer than 2 groups flashes an error and redirects back."""
        _setup_process(client, tmp_path)
        response = client.post(
            "/groups_to",
            data={"group_students[Klas A]": ["Jongen"]},
        )
        assert response.status_code == 302
        assert any(cat == "error" for cat, _ in _flashes(client))

    def test_post_two_groups_redirects_to_student_preferences(self, client, tmp_path):
        """POST /groups_to with ≥2 groups saves groups.xlsx and redirects to student_preferences."""
        _setup_process(client, tmp_path)
        response = client.post(
            "/groups_to",
            data={
                "group_students[Klas A]": ["Jongen", "Meisje"],
                "group_students[Klas B]": ["Jongen"],
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/student_preferences")


class TestNotTogetherPage:
    """Tests for POST /not_together skip and error paths."""

    def _mock_file_reads(self, monkeypatch):
        """Patch datareader calls so not_together_page does not need real xlsx files."""
        monkeypatch.setattr(
            flask_module.datareader,
            "read_groups_excel",
            lambda _: (
                {"Klas A": None, "Klas B": None},
                {"Klas A": "Klas A", "Klas B": "Klas B"},
            ),
        )
        mock_proc = MagicMock()
        mock_proc.get_students_meta_info.return_value = {"Alice": {}, "Bob": {}}
        monkeypatch.setattr(
            flask_module.datareader, "VoorkeurenProcessor", lambda _: mock_proc
        )

    def test_missing_files_flashes_error_and_redirects_to_student_preferences(
        self, client, tmp_path
    ):
        """not_together_page redirects gracefully when preferences.xlsx is missing."""
        _setup_process(client, tmp_path)
        response = client.get("/not_together")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/student_preferences")
        assert any(cat == "error" for cat, _ in _flashes(client))

    def test_post_skip_redirects_to_start_distribution(
        self, client, tmp_path, monkeypatch
    ):
        """POST /not_together with action=skip saves empty rules and redirects to start."""
        _setup_process(client, tmp_path)
        self._mock_file_reads(monkeypatch)
        response = client.post("/not_together", data={"action": "skip"})
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/start_distribution")

    def test_post_duplicate_student_flashes_error(self, client, tmp_path, monkeypatch):
        """A rule with the same student listed twice flashes a Dutch parse error."""
        _setup_process(client, tmp_path)
        self._mock_file_reads(monkeypatch)
        response = client.post(
            "/not_together",
            data={
                "n_rules": "1",
                "rule_students[0]": ["Alice", "Alice"],
                "rule_max[0]": "1",
            },
        )
        assert response.status_code == 302
        assert any(cat == "error" for cat, _ in _flashes(client))


class TestStartDistribution:
    """Tests for GET /start_distribution."""

    def _mock_threads(self, monkeypatch):
        """Patch the solver and sociogram so threads return immediately."""
        monkeypatch.setattr(
            flask_module, "distribute_students_once", MagicMock(return_value={})
        )
        monkeypatch.setattr(
            flask_module.datareader,
            "read_groups_excel",
            lambda _: ({"Klas A": None}, {"Klas A": "Klas A"}),
        )
        monkeypatch.setattr(flask_module.sociogram, "SociogramMaker", MagicMock())
        monkeypatch.setattr(flask_module.sociogram, "networkx_to_plotly", MagicMock())

    def test_happy_path_redirects_to_processing(self, client, tmp_path, monkeypatch):
        """start_distribution spawns threads and immediately redirects to /processing/<uuid>."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "preferences.xlsx").write_bytes(b"dummy")
        self._mock_threads(monkeypatch)
        response = client.get("/start_distribution")
        assert response.status_code == 302
        assert "/processing/" in response.headers["Location"]

    def test_not_together_json_is_loaded_when_present(
        self, client, tmp_path, monkeypatch
    ):
        """When not_together.json exists it is parsed and passed to distribute_students_once."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "preferences.xlsx").write_bytes(b"dummy")
        (proc_dir / "not_together.json").write_text(
            '[{"group": ["Alice", "Bob"], "Max_aantal_samen": 1}]', encoding="utf-8"
        )
        self._mock_threads(monkeypatch)
        response = client.get("/start_distribution")
        assert response.status_code == 302
        assert "/processing/" in response.headers["Location"]


class TestStatus:
    """Tests for GET /status/<task_id>."""

    def test_unknown_task_returns_unknown_status(self, client):
        """A task_id not in status_dct returns {"status_studentdistribution": "unknown"}."""
        response = client.get("/status/nonexistent-id")
        assert response.status_code == 200
        assert response.get_json()["status_studentdistribution"] == "unknown"

    def test_known_task_returns_stored_status(self, client, monkeypatch):
        """A task_id in status_dct returns the stored status dict as JSON."""
        monkeypatch.setitem(
            flask_module.status_dct,
            "abc123",
            {"status_studentdistribution": "done", "logs": []},
        )
        response = client.get("/status/abc123")
        assert response.get_json()["status_studentdistribution"] == "done"


class TestResultPage:
    """Tests for GET /result/<task_id>."""

    def test_unknown_task_flashes_and_redirects(self, client):
        """Visiting /result for an unknown task_id flashes an error and redirects to /processes."""
        response = client.get("/result/nonexistent")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")
        assert any(cat == "error" for cat, _ in _flashes(client))

    def test_incomplete_task_redirects(self, client, monkeypatch):
        """Visiting /result before groepsindeling is ready redirects cleanly."""
        monkeypatch.setitem(flask_module.temp_storage, "task1", {})
        response = client.get("/result/task1")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")


class TestSociogramPage:
    """Tests for GET /sociogram/<task_id>."""

    def test_unknown_task_flashes_and_redirects(self, client):
        """Visiting /sociogram for an unknown task_id flashes an error and redirects."""
        response = client.get("/sociogram/nonexistent")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_known_task_renders_sociogram(self, client, monkeypatch):
        """A task with a sociogram in temp_storage renders it in the page."""
        monkeypatch.setitem(
            flask_module.temp_storage, "task1", {"sociogram": "<div>plotly</div>"}
        )
        response = client.get("/sociogram/task1")
        assert response.status_code == 200
        assert b"plotly" in response.data


class TestDownload:
    """Tests for GET /download/<task_id>."""

    def test_unknown_task_renders_result_page_with_flash(self, client):
        """Downloading a non-existent result renders the result page with a Dutch error flash."""
        response = client.get("/download/nonexistent")
        assert response.status_code == 200
        # Flash is consumed by base.html during render; verify it appears in the HTML
        assert b"Groepsindeling niet gevonden" in response.data

    def test_ready_groepsindeling_sends_file(self, client, monkeypatch):
        """When groepsindeling is ready, the Excel file is sent as an attachment."""
        buf = BytesIO(b"dummy excel content")
        monkeypatch.setitem(
            flask_module.temp_storage,
            "task1",
            {"groepsindeling": {"download": buf}},
        )
        response = client.get("/download/task1")
        assert response.status_code == 200
        assert "attachment" in response.headers.get("Content-Disposition", "")


class TestHandleError:
    """Tests for POST /handle-error."""

    def test_valid_message_returns_204(self, client):
        """A valid JSON POST to /handle-error returns HTTP 204 No Content."""
        response = client.post(
            "/handle-error",
            json={"message": "Er ging iets mis"},
            content_type="application/json",
        )
        assert response.status_code == 204

    def test_valid_message_is_flashed(self, client):
        """The message from /handle-error is stored as a flash for the next request."""
        client.post(
            "/handle-error",
            json={"message": "Er ging iets mis"},
            content_type="application/json",
        )
        assert any(msg == "Er ging iets mis" for _, msg in _flashes(client))


class TestErrorMessages:
    """Unit tests for the user-facing Dutch error messages produced by app.py helpers.

    These functions are critical because they determine what Dutch text teachers see
    when their uploads fail. Tests here pin the Dutch strings so a refactor cannot
    silently change what the UI shows.
    """

    def test_unknown_exception_returns_generic_fallback(self):
        """A generic exception not in any known category returns the Dutch fallback."""
        msg = to_validation_message(RuntimeError("anything"))
        assert "onverwachts" in msg

    def test_readable_error_with_known_code_returns_dutch_template(self):
        """'wrong_columns_preferences' ValidationError returns the Dutch column-error template."""
        exc = ValidationError(
            "wrong_columns_preferences", {"wrong_columns": "Kolom A, Kolom B"}
        )
        msg = readableerror_to_validation_message(exc)
        assert "verkeerde kolommen" in msg
        assert "Kolom A, Kolom B" in msg

    def test_readable_error_with_unknown_code_returns_fallback(self):
        """A ValidationError with an unrecognised code falls back to the generic Dutch message."""
        exc = ValidationError("some_unknown_code", {})
        msg = readableerror_to_validation_message(exc)
        assert "onverwachts" in msg

    def test_too_few_students_not_together_returns_correct_dutch_text(self):
        """'too_few_students_not_together' error mentions the rule index and student minimum."""
        exc = ValidationError("too_few_students_not_together", {"rule_index": 2})
        msg = readableerror_to_validation_message(exc)
        assert "Niet-samen-regel 2" in msg
        assert "minstens 2 leerlingen" in msg

    def test_unknown_student_not_together_returns_student_name(self):
        """'unknown_student_not_together' error includes the unknown student names."""
        exc = ValidationError(
            "unknown_student_not_together", {"unknown_students": "Jan Jansen"}
        )
        msg = readableerror_to_validation_message(exc)
        assert "Jan Jansen" in msg
