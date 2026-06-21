"""Tests for routes/processes.py (processes blueprint)."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json

from werkzeug.security import generate_password_hash

import aliexpress.routes.processes as proc_module
from aliexpress.extensions import db
from aliexpress.models import Process, Run, School
from app import app as flask_app
from tests.helpers import SCHOOL_ID, flashes, make_process_row


class TestProcessesBlueprint:
    """Smoke tests: processes blueprint routes are reachable and require_process guards them."""

    def test_processes_list_is_reachable(self, client):
        """/processes returns 200 for an authenticated school user."""
        assert client.get("/processes").status_code == 200

    def test_require_process_redirects_without_active_process(self, client):
        """require_process redirects to /processes when no process is in session."""
        with client.session_transaction() as sess:
            sess.pop("process_id", None)
        resp = client.get("/groups_to")
        assert resp.status_code == 302
        assert "/processes" in resp.headers["Location"]

    def test_create_process_path_traversal_flashes_error(self, client, monkeypatch):
        """POST /processes/create with a traversal name shows a user-friendly error."""
        monkeypatch.setattr(
            proc_module,
            "get_process_path",
            lambda *_: (_ for _ in ()).throw(PermissionError("traversal")),
        )
        resp = client.post("/processes/create", data={"process_name": "ok-name"})
        assert resp.status_code == 302
        assert any(cat == "error" for cat, _ in flashes(client))


class TestCreateProcess:
    """Tests for POST /processes/create."""

    def test_empty_name_gives_naam_is_verplicht(self, client):
        """Bug 3: empty name must yield 'Naam is verplicht', not the regex message."""
        response = client.post("/processes/create", data={"process_name": ""})
        assert response.status_code == 302
        assert flashes(client) == [("error", "Naam is verplicht")]

    def test_invalid_chars_gives_format_error(self, client):
        """Invalid characters yield a format error."""
        response = client.post("/processes/create", data={"process_name": "bad/name!"})
        assert response.status_code == 302
        assert flashes(client) == [
            ("error", "Alleen letters, cijfers, spaties, - en _ toegestaan")
        ]

    def test_existing_name_gives_bestaat_al(self, client):
        """Bug 2: creating a duplicate must yield 'Proces bestaat al', not 'bestaat niet'."""
        with flask_app.app_context():
            make_process_row(SCHOOL_ID, "mijnproces")
        response = client.post("/processes/create", data={"process_name": "mijnproces"})
        assert response.status_code == 302
        assert flashes(client) == [("error", "Proces bestaat al")]

    def test_happy_path_creates_directory(self, client, tmp_path):
        """A valid new name creates the process directory and redirects to upload."""
        response = client.post(
            "/processes/create", data={"process_name": "nieuwproces"}
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/upload_edexml")
        assert (tmp_path / SCHOOL_ID / "nieuwproces").is_dir()


class TestDeleteProcess:
    """Tests for POST /processes/delete/<process_name>."""

    def test_nonexistent_name_gives_bestaat_niet(self, client):
        """Bug 2: deleting a missing process must yield 'Proces bestaat niet', not 'bestaat al'."""
        response = client.post("/processes/delete/spookproces")
        assert response.status_code == 302
        assert flashes(client) == [("error", "Proces bestaat niet")]

    def test_invalid_chars_gives_format_error(self, client):
        """A name with a slash hits the router before validation; expect 302 or 404."""
        response = client.post("/processes/delete/bad/name")
        assert response.status_code in (302, 404)

    def test_happy_path_removes_directory(self, client, tmp_path):
        """Deleting an existing process removes the directory and redirects."""
        proc_dir = tmp_path / SCHOOL_ID / "teproces"
        proc_dir.mkdir(parents=True, exist_ok=True)
        with flask_app.app_context():
            make_process_row(SCHOOL_ID, "teproces")
        response = client.post("/processes/delete/teproces")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")
        assert not proc_dir.exists()


class TestProcessesList:
    """Tests for GET /processes."""

    def test_empty_dir_returns_200(self, client):
        """An empty BASE_DIR produces an empty process list without errors."""
        assert client.get("/processes").status_code == 200

    def test_existing_process_is_shown(self, client):
        """A process that exists in the DB appears in the processes list."""
        with flask_app.app_context():
            make_process_row(SCHOOL_ID, "mijnklas")
        response = client.get("/processes")
        assert response.status_code == 200
        assert b"mijnklas" in response.data


class TestSelectProcess:
    """Tests for GET /processes/select/<process_id>."""

    def test_unknown_process_redirects_with_error(self, client):
        """Selecting a process that does not exist flashes an error and redirects."""
        response = client.get("/processes/select/bestaat_niet")
        assert response.status_code == 302
        assert flashes(client) == [
            ("error", "Deze pagina bestaat niet of je hebt er geen toegang toe.")
        ]

    def test_malformed_process_id_redirects_with_error(self, client, tmp_path):
        """A tampered id with path characters is rejected on format, before any path use.

        ``bad.name`` would reach the route (dots are valid in a URL segment) but must not
        be turned into a filesystem path: the format check rejects it and redirects.
        """
        # Even if a dir existed, the format check must block it.
        (tmp_path / SCHOOL_ID / "bad.name").mkdir(parents=True, exist_ok=True)
        response = client.get("/processes/select/bad.name")
        assert response.status_code == 302
        assert flashes(client) == [
            ("error", "Deze pagina bestaat niet of je hebt er geen toegang toe.")
        ]

    def test_empty_process_redirects_to_upload_edexml(self, client, tmp_path):
        """A process with no files starts at the first step: upload EDEXML."""
        (tmp_path / SCHOOL_ID / "leegproces").mkdir(parents=True, exist_ok=True)
        with flask_app.app_context():
            make_process_row(SCHOOL_ID, "leegproces")
        response = client.get("/processes/select/leegproces")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/upload_edexml")

    def test_process_with_json_redirects_to_groups_to(self, client, tmp_path):
        """A process that has the candidates JSON continues at groups_to."""
        proc_dir = tmp_path / SCHOOL_ID / "procesmetjson"
        proc_dir.mkdir(parents=True, exist_ok=True)
        (proc_dir / "relevant_students_and_groups.json").write_text(
            "{}", encoding="utf-8"
        )
        with flask_app.app_context():
            make_process_row(SCHOOL_ID, "procesmetjson")
        response = client.get("/processes/select/procesmetjson")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/groups_to")

    def test_process_with_groups_xlsx_redirects_to_roster(self, client, tmp_path):
        """A process that has groups.xlsx but no roster yet continues at the shared
        "Wie gaat mee" step (ADR 0005)."""
        proc_dir = tmp_path / SCHOOL_ID / "procesmetgroepen"
        proc_dir.mkdir(parents=True, exist_ok=True)
        (proc_dir / "groups.xlsx").write_bytes(b"dummy")
        with flask_app.app_context():
            make_process_row(SCHOOL_ID, "procesmetgroepen")
        response = client.get("/processes/select/procesmetgroepen")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/roster")

    def test_process_with_roster_and_excel_method_redirects_to_preferences_excel(
        self, client, tmp_path
    ):
        """Once the roster is settled, an Excel-method process resumes at preferences_excel."""
        proc_dir = tmp_path / SCHOOL_ID / "procesmetroster"
        proc_dir.mkdir(parents=True, exist_ok=True)
        (proc_dir / "groups.xlsx").write_bytes(b"dummy")
        (proc_dir / "roster.json").write_text(
            json.dumps({"participants": []}), encoding="utf-8"
        )
        (proc_dir / "input_method.json").write_text(
            json.dumps({"method": "excel"}), encoding="utf-8"
        )
        with flask_app.app_context():
            make_process_row(SCHOOL_ID, "procesmetroster")
        response = client.get("/processes/select/procesmetroster")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_excel")

    def test_process_with_roster_and_form_method_redirects_to_preferences_form(
        self, client, tmp_path
    ):
        """A process with a settled roster + input_method form resumes at preferences_form."""
        proc_dir = tmp_path / SCHOOL_ID / "procesformulier"
        proc_dir.mkdir(parents=True, exist_ok=True)
        (proc_dir / "groups.xlsx").write_bytes(b"dummy")
        (proc_dir / "roster.json").write_text(
            json.dumps({"participants": []}), encoding="utf-8"
        )
        (proc_dir / "input_method.json").write_text(
            json.dumps({"method": "form"}), encoding="utf-8"
        )
        with flask_app.app_context():
            make_process_row(SCHOOL_ID, "procesformulier")
        response = client.get("/processes/select/procesformulier")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_form")

    def test_process_with_preferences_xlsx_redirects_to_not_together(
        self, client, tmp_path
    ):
        """A process that has preferences.xlsx continues at not_together."""
        proc_dir = tmp_path / SCHOOL_ID / "procesmetpref"
        proc_dir.mkdir(parents=True, exist_ok=True)
        (proc_dir / "preferences.xlsx").write_bytes(b"dummy")
        with flask_app.app_context():
            make_process_row(SCHOOL_ID, "procesmetpref")
        response = client.get("/processes/select/procesmetpref")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/not_together")

    def test_process_with_voorkeuren_json_redirects_to_not_together(
        self, client, tmp_path
    ):
        """A process that has only voorkeuren.json (form path) continues at not_together."""
        proc_dir = tmp_path / SCHOOL_ID / "procesmetjson"
        proc_dir.mkdir(parents=True, exist_ok=True)
        (proc_dir / "voorkeuren.json").write_text("{}", encoding="utf-8")
        with flask_app.app_context():
            make_process_row(SCHOOL_ID, "procesmetjson")
        response = client.get("/processes/select/procesmetjson")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/not_together")

    def test_select_with_done_run_redirects_to_result(self, client, tmp_path):
        """A completed run skips the wizard and goes straight to the result page."""
        proc_dir = tmp_path / SCHOOL_ID / "gedaanproces"
        proc_dir.mkdir(parents=True, exist_ok=True)
        with flask_app.app_context():
            proc = make_process_row(SCHOOL_ID, "gedaanproces")
            run = Run(process_id=proc.id, status="done")
            db.session.add(run)
            db.session.commit()
        response = client.get("/processes/select/gedaanproces")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/result")

    def test_select_with_running_run_redirects_to_processing(self, client, tmp_path):
        """An in-progress run resumes at the processing/status page."""
        proc_dir = tmp_path / SCHOOL_ID / "looptproces"
        proc_dir.mkdir(parents=True, exist_ok=True)
        with flask_app.app_context():
            proc = make_process_row(SCHOOL_ID, "looptproces")
            run = Run(process_id=proc.id, status="running")
            db.session.add(run)
            db.session.commit()
        response = client.get("/processes/select/looptproces")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processing")

    def test_select_with_error_run_redirects_to_processing(self, client, tmp_path):
        """A failed run reopens the processing page so the user sees the error."""
        proc_dir = tmp_path / SCHOOL_ID / "foutproces"
        proc_dir.mkdir(parents=True, exist_ok=True)
        with flask_app.app_context():
            proc = make_process_row(SCHOOL_ID, "foutproces")
            run = Run(process_id=proc.id, status="error")
            db.session.add(run)
            db.session.commit()
        response = client.get("/processes/select/foutproces")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processing")


class TestSchoolIsolation:
    """A school cannot access, see, or delete another school's processes."""

    def _create_other_school_process(self, process_name="geheimproces"):
        """Insert a school + process that the test-school must not be able to reach."""
        with flask_app.app_context():
            andere = School(
                schoolcode="andere-school",
                naam="Andere School",
                password_hash=generate_password_hash("pw"),
            )
            db.session.add(andere)
            proc = Process(school_id="andere-school", name=process_name)
            db.session.add(proc)
            db.session.commit()

    def test_cannot_select_other_schools_process(self, client):
        """GET /processes/select/<id> flashes an error for a process owned by another school."""
        self._create_other_school_process()
        response = client.get("/processes/select/geheimproces")
        assert response.status_code == 302
        assert flashes(client) == [
            ("error", "Deze pagina bestaat niet of je hebt er geen toegang toe.")
        ]

    def test_cannot_delete_other_schools_process(self, client):
        """POST /processes/delete/<name> flashes 'bestaat niet' for another school's process."""
        self._create_other_school_process()
        response = client.post("/processes/delete/geheimproces")
        assert response.status_code == 302
        assert flashes(client) == [("error", "Proces bestaat niet")]

    def test_cannot_see_other_schools_process_in_list(self, client):
        """GET /processes does not include processes from other schools."""
        self._create_other_school_process()
        response = client.get("/processes")
        assert response.status_code == 200
        assert b"geheimproces" not in response.data
