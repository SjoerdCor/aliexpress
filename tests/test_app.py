"""Tests for app.py Flask routes."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
import re
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from werkzeug.datastructures import MultiDict

import app as flask_module
from aliexpress.errors import ValidationError
from aliexpress.extensions import db
from aliexpress.models import LogLine, Run
from app import app as flask_app


def _immediate_thread(target, args=()):
    """Thread replacement whose ``start()`` runs the target synchronously, so route-spawned
    background work finishes before the request returns and is deterministic to assert on.
    """
    runner = MagicMock()
    runner.start.side_effect = lambda: target(*args)
    return runner


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

    def test_processing_returns_200(self, client, tmp_path):
        """GET /processing renders the processing page for the active process."""
        _setup_process(client, tmp_path)
        assert client.get("/processing").status_code == 200


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

    def test_malformed_process_id_gives_404(self, client, tmp_path):
        """A tampered id with path characters is rejected on format, before any path use.

        ``bad.name`` would reach the route (dots are valid in a URL segment) but must not
        be turned into a filesystem path: the format check rejects it with a 404.
        """
        (
            tmp_path / "bad.name"
        ).mkdir()  # even if such a dir existed, it must not be served
        assert client.get("/processes/select/bad.name").status_code == 404

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


def _write_groups_to_json(proc_dir, groups_to):
    """Persist a candidates JSON whose groups_to maps each group to student dicts."""
    (proc_dir / "relevant_students_and_groups.json").write_text(
        json.dumps({"groups_to": groups_to}), encoding="utf-8"
    )


def _g(*genders):
    """Build a list of minimal student dicts with the given genders, in order."""
    return [{"geslacht": sex, "roepnaam": "x", "achternaam": "y"} for sex in genders]


class TestGroupsToPage:
    """Tests for GET/POST /groups_to."""

    def test_get_renders_groups_from_json(self, client, tmp_path):
        """GET /groups_to reads the candidates JSON and renders group names in the page."""
        proc_dir = _setup_process(client, tmp_path)
        # groups_to is a dict {groupname: [students]}; the template calls .items()
        _write_groups_to_json(proc_dir, {"Klas A": [], "Klas B": []})
        response = client.get("/groups_to")
        assert response.status_code == 200
        assert b"Klas A" in response.data

    def test_post_too_few_groups_flashes_error(self, client, tmp_path):
        """POST /groups_to with fewer than 2 groups flashes an error and redirects back."""
        proc_dir = _setup_process(client, tmp_path)
        _write_groups_to_json(proc_dir, {"Klas A": _g("Jongen")})
        response = client.post(
            "/groups_to",
            data={"group": ["Klas A"], "group_students[Klas A]": ["0"]},
        )
        assert response.status_code == 302
        assert any(cat == "error" for cat, _ in _flashes(client))

    def test_post_two_groups_redirects_to_student_preferences(self, client, tmp_path):
        """POST /groups_to with ≥2 groups saves groups.xlsx and redirects to student_preferences."""
        proc_dir = _setup_process(client, tmp_path)
        _write_groups_to_json(
            proc_dir, {"Klas A": _g("Jongen", "Meisje"), "Klas B": _g("Jongen")}
        )
        response = client.post(
            "/groups_to",
            data={
                "group": ["Klas A", "Klas B"],
                "group_students[Klas A]": ["0", "1"],
                "group_students[Klas B]": ["0"],
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/student_preferences")

    def test_post_empty_group_is_kept_with_zero_counts(self, client, tmp_path):
        """A group submitted via 'group' but without retained students is kept at 0/0."""
        proc_dir = _setup_process(client, tmp_path)
        _write_groups_to_json(proc_dir, {"Klas A": _g("Jongen", "Meisje", "Meisje")})
        response = client.post(
            "/groups_to",
            data={
                "group": ["Klas A", "Nieuwe groep 1"],
                "group_students[Klas A]": ["0", "1", "2"],
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/student_preferences")
        saved = pd.read_excel(proc_dir / "groups.xlsx", index_col=0)
        assert saved.loc["Klas A", "Jongens"] == 1
        assert saved.loc["Klas A", "Meisjes"] == 2
        assert saved.loc["Nieuwe groep 1", "Jongens"] == 0
        assert saved.loc["Nieuwe groep 1", "Meisjes"] == 0

    def test_post_persists_restore_state(self, client, tmp_path):
        """POST writes groups_to_state.json capturing ticks, disabled and new groups."""
        proc_dir = _setup_process(client, tmp_path)
        _write_groups_to_json(
            proc_dir, {"Klas A": _g("Jongen", "Meisje"), "Klas B": _g("Jongen")}
        )
        client.post(
            "/groups_to",
            data={
                # Klas B switched off (absent from 'group'); a new empty group added.
                "group": ["Klas A", "Nieuwe groep 1"],
                "group_students[Klas A]": ["1"],
            },
        )
        state = json.loads((proc_dir / "groups_to_state.json").read_text("utf-8"))
        assert state["original_groups"]["Klas A"]["checked_indices"] == [1]
        assert state["disabled_groups"] == ["Klas B"]
        assert state["new_groups"] == ["Nieuwe groep 1"]

    def test_get_restores_state_into_the_form(self, client, tmp_path):
        """GET after a save pre-ticks the right boxes and marks a disabled group."""
        proc_dir = _setup_process(client, tmp_path)
        _write_groups_to_json(
            proc_dir, {"Klas A": _g("Jongen", "Meisje"), "Klas B": _g("Jongen")}
        )
        (proc_dir / "groups_to_state.json").write_text(
            json.dumps(
                {
                    "original_groups": {"Klas A": {"checked_indices": [1]}},
                    "disabled_groups": ["Klas B"],
                    "new_groups": ["Nieuwe groep 1"],
                }
            ),
            encoding="utf-8",
        )
        html = client.get("/groups_to").data.decode("utf-8")
        checkboxes = re.findall(r'<input type="checkbox".*?>', html, re.DOTALL)
        ticked = [c for c in checkboxes if "checked" in c]
        # Exactly one box is ticked: the second student of Klas A (index 1).
        assert len(ticked) == 1
        assert 'value="1"' in ticked[0]
        assert "group-disabled" in html  # Klas B comes in switched off
        assert "Nieuwe groep 1" in html  # restored new group is rendered


class TestParseGroupsToForm:
    """Tests for the form-parsing helper parse_groups_to_form."""

    def test_counts_genders_and_keeps_empty_group(self):
        """Genders are looked up by index; a submitted group without ticks stays 0/0."""
        groups_to = {"Klas A": _g("Jongen", "Meisje", "Jongen"), "Klas B": _g("Jongen")}
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group", "Klas B"),
                ("group_students[Klas A]", "0"),
                ("group_students[Klas A]", "1"),
                ("group_students[Klas A]", "2"),
            ]
        )
        result = flask_module.parse_groups_to_form(form, groups_to)
        assert result.distribution == {
            "Klas A": {"Jongens": 2, "Meisjes": 1},
            "Klas B": {"Jongens": 0, "Meisjes": 0},
        }
        assert result.state["original_groups"]["Klas A"]["checked_indices"] == [0, 1, 2]
        assert result.state["disabled_groups"] == []
        assert result.state["new_groups"] == []

    def test_disabled_and_new_groups_are_recorded(self):
        """An original group absent from 'group' is disabled; an unknown name is new."""
        groups_to = {"Klas A": _g("Jongen", "Meisje"), "Klas B": _g("Jongen")}
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group", "Nieuwe groep 1"),
                ("group_students[Klas A]", "0"),
            ]
        )
        result = flask_module.parse_groups_to_form(form, groups_to)
        assert result.state["disabled_groups"] == ["Klas B"]
        assert result.state["new_groups"] == ["Nieuwe groep 1"]
        assert result.distribution["Nieuwe groep 1"] == {"Jongens": 0, "Meisjes": 0}

    def test_switched_off_group_keeps_its_ticks(self):
        """A switched-off group still submits its boxes, so its ticks are remembered."""
        groups_to = {"Klas A": _g("Jongen"), "Klas B": _g("Jongen", "Meisje")}
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group_students[Klas A]", "0"),
                # Klas B is switched off (absent from 'group') but its boxes still submit.
                ("group_students[Klas B]", "1"),
            ]
        )
        result = flask_module.parse_groups_to_form(form, groups_to)
        assert result.state["disabled_groups"] == ["Klas B"]
        assert result.state["original_groups"]["Klas B"]["checked_indices"] == [1]
        # Switched-off groups must not reach groups.xlsx.
        assert "Klas B" not in result.distribution

    def test_out_of_range_or_non_numeric_indices_are_ignored(self):
        """Tampered indices that fall outside the student list are dropped safely."""
        groups_to = {"Klas A": _g("Jongen")}
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group_students[Klas A]", "0"),
                ("group_students[Klas A]", "9"),
                ("group_students[Klas A]", "x"),
            ]
        )
        result = flask_module.parse_groups_to_form(form, groups_to)
        assert result.distribution["Klas A"] == {"Jongens": 1, "Meisjes": 0}
        assert result.state["original_groups"]["Klas A"]["checked_indices"] == [0]


class TestStudentPreferencesSelection:
    """GET /student_preferences restores the Stap 1 selection saved on download."""

    CANDIDATES = [
        {"key": "s1", "roepnaam": "Anna", "achternaam": "Bos", "groepsnaam": "Groen"},
        {"key": "s2", "roepnaam": "Bram", "achternaam": "Dijk", "groepsnaam": "Groen"},
        {"key": "s3", "roepnaam": "Cas", "achternaam": "El", "groepsnaam": "Blauw"},
    ]

    def _setup(self, client, tmp_path):
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "relevant_students_and_groups.json").write_text(
            json.dumps(
                {"candidates": self.CANDIDATES, "groups_from": ["Groen", "Blauw"]}
            ),
            encoding="utf-8",
        )
        return proc_dir

    def _checkbox_state(self, html):
        """Map each candidate key to whether its checkbox is ticked."""
        boxes = re.findall(r'<input type="checkbox" name="students"[^>]*>', html)
        return {re.search(r'value="(s\d)"', b).group(1): "checked" in b for b in boxes}

    def test_first_visit_ticks_all_candidates(self, client, tmp_path):
        """Without a saved selection every candidate starts ticked."""
        self._setup(client, tmp_path)
        html = client.get("/student_preferences").data.decode("utf-8")
        assert self._checkbox_state(html) == {"s1": True, "s2": True, "s3": True}

    def test_saved_selection_and_added_student_are_restored(self, client, tmp_path):
        """A saved selection un-ticks dropped students and re-fills added ones."""
        proc_dir = self._setup(client, tmp_path)
        (proc_dir / "student_selection.json").write_text(
            json.dumps(
                {
                    "selected_ids": ["s1", "s3"],
                    "new_students": [
                        {
                            "roepnaam": "Daan",
                            "achternaam": "Fok",
                            "geslacht": "Jongen",
                            "groepsnaam": "Blauw",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        html = client.get("/student_preferences").data.decode("utf-8")
        assert self._checkbox_state(html) == {"s1": True, "s2": False, "s3": True}
        assert 'value="Daan"' in html
        assert 'value="Jongen" selected' in html
        assert 'value="Blauw" selected' in html


class TestNotTogetherPage:
    """Tests for POST /not_together error paths."""

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
        # The dropdown lists the names as entered (display map values).
        mock_proc.student_display = {"alice": "Alice", "bob": "Bob"}
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
    """Tests for GET /start_distribution (run lifecycle with a mocked solver)."""

    def _patch_pipeline(self, monkeypatch, *, result=None, exc=None):
        """Run both background threads synchronously with a mocked solver and sociogram.

        Returns the solver mock so a test can inspect the arguments it was called with.
        """
        monkeypatch.setattr(flask_module, "Thread", _immediate_thread)
        solver = MagicMock(side_effect=exc) if exc else MagicMock(return_value=result)
        monkeypatch.setattr(flask_module, "distribute_students_once", solver)
        monkeypatch.setattr(
            flask_module.datareader,
            "read_groups_excel",
            lambda _: ({"Klas A": None}, {"Klas A": "Klas A"}),
        )
        maker = MagicMock()
        maker.plot_sociogram.return_value = (MagicMock(), MagicMock(), MagicMock())
        monkeypatch.setattr(
            flask_module.sociogram, "SociogramMaker", lambda *a, **k: maker
        )
        fig = MagicMock()
        fig.to_html.return_value = "<div>socio</div>"
        monkeypatch.setattr(
            flask_module.sociogram, "networkx_to_plotly", lambda *a, **k: fig
        )
        return solver

    def _read_run(self, process_id="testproces"):
        with flask_app.app_context():
            return db.session.get(Run, process_id)

    def test_happy_path_writes_files_and_marks_done(
        self, client, tmp_path, monkeypatch
    ):
        """A successful run writes the artifacts and only then sets status 'done'."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "preferences.xlsx").write_bytes(b"dummy")
        (proc_dir / "groups.xlsx").write_bytes(b"dummy")
        result = {
            "download": BytesIO(b"excel-bytes"),
            "dataframes": {"Groepsindeling": pd.DataFrame({"A": [1]})},
        }
        self._patch_pipeline(monkeypatch, result=result)

        response = client.get("/start_distribution")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processing")
        assert (proc_dir / "results.xlsx").read_bytes() == b"excel-bytes"
        tables = json.loads((proc_dir / "result_tables.json").read_text("utf-8"))
        assert "Groepsindeling" in tables
        assert (proc_dir / "sociogram.html").read_text("utf-8") == "<div>socio</div>"
        assert self._read_run().status == "done"

    def test_error_path_sets_error_status_and_message(
        self, client, tmp_path, monkeypatch
    ):
        """A solver failure records status 'error' with a friendly message, no result file."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "preferences.xlsx").write_bytes(b"dummy")
        (proc_dir / "groups.xlsx").write_bytes(b"dummy")
        exc = ValidationError("wrong_columns_preferences", {"wrong_columns": "Kolom A"})
        self._patch_pipeline(monkeypatch, exc=exc)

        client.get("/start_distribution")
        run = self._read_run()
        assert run.status == "error"
        assert "verkeerde kolommen" in run.message
        assert not (proc_dir / "results.xlsx").exists()

    def test_not_together_json_is_loaded_when_present(
        self, client, tmp_path, monkeypatch
    ):
        """When not_together.json exists it is parsed and passed to distribute_students_once."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "preferences.xlsx").write_bytes(b"dummy")
        (proc_dir / "groups.xlsx").write_bytes(b"dummy")
        (proc_dir / "not_together.json").write_text(
            '[{"group": ["Alice", "Bob"], "Max_aantal_samen": 1}]', encoding="utf-8"
        )
        result = {"download": BytesIO(b"x"), "dataframes": {}}
        solver = self._patch_pipeline(monkeypatch, result=result)

        response = client.get("/start_distribution")
        assert response.status_code == 302
        passed = solver.call_args.args[2]
        assert passed == [{"group": {"Alice", "Bob"}, "Max_aantal_samen": 1}]


class TestStatus:
    """Tests for GET /status (process-scoped)."""

    def test_no_session_redirects(self, client):
        """Without an active process /status redirects to /processes."""
        response = client.get("/status")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_no_run_returns_unknown_status(self, client, tmp_path):
        """A process without a run row reports status 'unknown' and empty logs."""
        _setup_process(client, tmp_path)
        data = client.get("/status").get_json()
        assert data["status_studentdistribution"] == "unknown"
        assert data["logs"] == []

    def test_running_run_returns_status_and_logs(self, client, tmp_path):
        """A running run returns its status and log lines in insertion order."""
        _setup_process(client, tmp_path)
        with flask_app.app_context():
            db.session.add(Run(process_id="testproces", status="running"))
            db.session.add(LogLine(process_id="testproces", text="Eerste"))
            db.session.add(LogLine(process_id="testproces", text="Tweede"))
            db.session.commit()
        data = client.get("/status").get_json()
        assert data["status_studentdistribution"] == "running"
        assert data["logs"] == ["Eerste", "Tweede"]

    def test_error_run_includes_message(self, client, tmp_path):
        """An errored run exposes its friendly message for the frontend to flash."""
        _setup_process(client, tmp_path)
        with flask_app.app_context():
            db.session.add(
                Run(process_id="testproces", status="error", message="Mislukt")
            )
            db.session.commit()
        data = client.get("/status").get_json()
        assert data["status_studentdistribution"] == "error"
        assert data["message"] == "Mislukt"


class TestResultPage:
    """Tests for GET /result (process-scoped)."""

    def test_no_session_redirects(self, client):
        """Without an active process /result redirects to /processes."""
        response = client.get("/result")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_missing_tables_flashes_and_redirects(self, client, tmp_path):
        """Visiting /result before the tables file exists flashes an error and redirects."""
        _setup_process(client, tmp_path)
        response = client.get("/result")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")
        assert any(cat == "error" for cat, _ in _flashes(client))

    def test_renders_tables_from_file(self, client, tmp_path):
        """The result page renders the stored HTML tables."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "result_tables.json").write_text(
            json.dumps({"Groepsindeling": "<table>indeling</table>"}),
            encoding="utf-8",
        )
        html = client.get("/result").data.decode("utf-8")
        assert "Groepsindeling" in html
        assert "<table>indeling</table>" in html


class TestSociogramPage:
    """Tests for GET /sociogram (process-scoped)."""

    def test_no_session_redirects(self, client):
        """Without an active process /sociogram redirects to /processes."""
        response = client.get("/sociogram")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_missing_file_flashes_and_redirects(self, client, tmp_path):
        """Visiting /sociogram before the file exists flashes an error and redirects."""
        _setup_process(client, tmp_path)
        response = client.get("/sociogram")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_renders_sociogram_file(self, client, tmp_path):
        """A stored sociogram.html is rendered into the page."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "sociogram.html").write_text("<div>plotly</div>", encoding="utf-8")
        response = client.get("/sociogram")
        assert response.status_code == 200
        assert b"plotly" in response.data


class TestDownload:
    """Tests for GET /download (process-scoped)."""

    def test_no_session_redirects(self, client):
        """Without an active process /download redirects to /processes."""
        response = client.get("/download")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_missing_file_renders_result_page_with_flash(self, client, tmp_path):
        """Downloading before the result file exists renders the result page with a flash."""
        _setup_process(client, tmp_path)
        response = client.get("/download")
        assert response.status_code == 200
        # Flash is consumed by base.html during render; verify it appears in the HTML
        assert b"Groepsindeling niet gevonden" in response.data

    def test_existing_file_sends_attachment(self, client, tmp_path):
        """When results.xlsx exists it is sent as an attachment."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "results.xlsx").write_bytes(b"dummy excel content")
        response = client.get("/download")
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


class TestSecretKeyGuard:
    """The startup guard refuses to run without a SECRET_KEY."""

    def test_missing_secret_key_raises(self):
        """An empty SECRET_KEY must raise at startup, not silently run unsigned."""
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
        _setup_process(client, tmp_path)
        monkeypatch.setitem(client.application.config, "MAX_CONTENT_LENGTH", 50)
        response = client.post(
            "/upload_edexml",
            data={"edexml": (BytesIO(b"x" * 5000), "edex.xml"), "jaargroep": "4"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 302
        assert any(
            cat == "error" and "te groot" in msg for cat, msg in _flashes(client)
        )
