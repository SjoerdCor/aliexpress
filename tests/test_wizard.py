"""Tests for routes/wizard.py (wizard blueprint)."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
import re
from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd
from werkzeug.datastructures import MultiDict

import aliexpress.routes.wizard as wizard_module
from aliexpress.errors import ValidationError
from aliexpress.extensions import db
from aliexpress.models import LogLine, Process, Run
from app import app as flask_app
from tests.helpers import (
    SCHOOL_ID,
    flashes,
    immediate_thread,
    make_students,
    setup_process,
    write_groups_to_json,
    write_minimal_voorkeuren_json,
)


class TestUploadErrors:
    """Tests for friendly error handling on file upload routes."""

    def test_garbage_preferences_redirects_with_error_flash(
        self, client, tmp_path, monkeypatch
    ):
        """Uploading a garbage file as preferences flashes an error and redirects back."""
        proc_dir = setup_process(client, tmp_path)
        monkeypatch.setattr(
            wizard_module.datareader,
            "read_groups_excel",
            lambda _path: ({"Klas A": None}, {"Klas A": "Klas A"}),
        )

        response = client.post(
            "/upload_preferences",
            data={"preferences": (BytesIO(b"not an excel"), "voorkeuren.xlsx")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_excel")
        assert any(cat == "error" for cat, _msg in flashes(client))
        assert not (proc_dir / "preferences.xlsx").exists()

    def test_wrong_column_preferences_flashes_column_message(
        self, client, tmp_path, monkeypatch
    ):
        """An Excel with wrong columns produces the column-mismatch Dutch message."""
        setup_process(client, tmp_path)
        monkeypatch.setattr(
            wizard_module.datareader,
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
        assert response.headers["Location"].endswith("/preferences_excel")
        assert any(
            cat == "error" and "verkeerde kolommen" in msg
            for cat, msg in flashes(client)
        )

    def test_garbage_edexml_redirects_with_error_flash(self, client, tmp_path):
        """Uploading a garbage EDEXML file flashes an error and redirects back."""
        setup_process(client, tmp_path)

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
        assert any(cat == "error" for cat, _msg in flashes(client))


class TestGroupsToPage:
    """Tests for GET/POST /groups_to."""

    def test_get_renders_groups_from_json(self, client, tmp_path):
        """GET /groups_to reads the candidates JSON and renders group names in the page."""
        proc_dir = setup_process(client, tmp_path)
        # groups_to is a dict {groupname: [students]}; the template calls .items()
        write_groups_to_json(proc_dir, {"Klas A": [], "Klas B": []})
        response = client.get("/groups_to")
        assert response.status_code == 200
        assert b"Klas A" in response.data

    def test_post_too_few_groups_flashes_error(self, client, tmp_path):
        """POST /groups_to with fewer than 2 groups flashes an error and redirects back."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(proc_dir, {"Klas A": make_students("Jongen")})
        response = client.post(
            "/groups_to",
            data={"group": ["Klas A"], "group_students[Klas A]": ["0"]},
        )
        assert response.status_code == 302
        assert any(cat == "error" for cat, _ in flashes(client))

    def test_post_two_groups_redirects_to_preferences_excel(self, client, tmp_path):
        """POST /groups_to with ≥2 groups saves groups.xlsx and redirects to preferences_excel."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(
            proc_dir,
            {
                "Klas A": make_students("Jongen", "Meisje"),
                "Klas B": make_students("Jongen"),
            },
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
        assert response.headers["Location"].endswith("/preferences_excel")

    def test_post_two_groups_with_form_action_redirects_to_preferences_form(
        self, client, tmp_path
    ):
        """POST /groups_to with action='formulier' redirects to /preferences_form."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(
            proc_dir,
            {
                "Klas A": make_students("Jongen", "Meisje"),
                "Klas B": make_students("Jongen"),
            },
        )
        response = client.post(
            "/groups_to",
            data={
                "group": ["Klas A", "Klas B"],
                "group_students[Klas A]": ["0", "1"],
                "group_students[Klas B]": ["0"],
                "action": "form",
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_form")

    def test_post_empty_group_is_kept_with_zero_counts(self, client, tmp_path):
        """A group submitted via 'group' but without retained students is kept at 0/0."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(
            proc_dir, {"Klas A": make_students("Jongen", "Meisje", "Meisje")}
        )
        response = client.post(
            "/groups_to",
            data={
                "group": ["Klas A", "Nieuwe groep 1"],
                "group_students[Klas A]": ["0", "1", "2"],
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_excel")
        saved = pd.read_excel(proc_dir / "groups.xlsx", index_col=0)
        assert saved.loc["Klas A", "Jongens"] == 1
        assert saved.loc["Klas A", "Meisjes"] == 2
        assert saved.loc["Nieuwe groep 1", "Jongens"] == 0
        assert saved.loc["Nieuwe groep 1", "Meisjes"] == 0

    def test_post_persists_restore_state(self, client, tmp_path):
        """POST writes groups_to_state.json capturing ticks, disabled and new groups."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(
            proc_dir,
            {
                "Klas A": make_students("Jongen", "Meisje"),
                "Klas B": make_students("Jongen"),
            },
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
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(
            proc_dir,
            {
                "Klas A": make_students("Jongen", "Meisje"),
                "Klas B": make_students("Jongen"),
            },
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
        groups_to = {
            "Klas A": make_students("Jongen", "Meisje", "Jongen"),
            "Klas B": make_students("Jongen"),
        }
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group", "Klas B"),
                ("group_students[Klas A]", "0"),
                ("group_students[Klas A]", "1"),
                ("group_students[Klas A]", "2"),
            ]
        )
        result = wizard_module.parse_groups_to_form(form, groups_to)
        assert result.distribution == {
            "Klas A": {"Jongens": 2, "Meisjes": 1},
            "Klas B": {"Jongens": 0, "Meisjes": 0},
        }
        assert result.state["original_groups"]["Klas A"]["checked_indices"] == [0, 1, 2]
        assert result.state["disabled_groups"] == []
        assert result.state["new_groups"] == []

    def test_disabled_and_new_groups_are_recorded(self):
        """An original group absent from 'group' is disabled; an unknown name is new."""
        groups_to = {
            "Klas A": make_students("Jongen", "Meisje"),
            "Klas B": make_students("Jongen"),
        }
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group", "Nieuwe groep 1"),
                ("group_students[Klas A]", "0"),
            ]
        )
        result = wizard_module.parse_groups_to_form(form, groups_to)
        assert result.state["disabled_groups"] == ["Klas B"]
        assert result.state["new_groups"] == ["Nieuwe groep 1"]
        assert result.distribution["Nieuwe groep 1"] == {"Jongens": 0, "Meisjes": 0}

    def test_switched_off_group_keeps_its_ticks(self):
        """A switched-off group still submits its boxes, so its ticks are remembered."""
        groups_to = {
            "Klas A": make_students("Jongen"),
            "Klas B": make_students("Jongen", "Meisje"),
        }
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group_students[Klas A]", "0"),
                # Klas B is switched off (absent from 'group') but its boxes still submit.
                ("group_students[Klas B]", "1"),
            ]
        )
        result = wizard_module.parse_groups_to_form(form, groups_to)
        assert result.state["disabled_groups"] == ["Klas B"]
        assert result.state["original_groups"]["Klas B"]["checked_indices"] == [1]
        # Switched-off groups must not reach groups.xlsx.
        assert "Klas B" not in result.distribution

    def test_out_of_range_or_non_numeric_indices_are_ignored(self):
        """Tampered indices that fall outside the student list are dropped safely."""
        groups_to = {"Klas A": make_students("Jongen")}
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group_students[Klas A]", "0"),
                ("group_students[Klas A]", "9"),
                ("group_students[Klas A]", "x"),
            ]
        )
        result = wizard_module.parse_groups_to_form(form, groups_to)
        assert result.distribution["Klas A"] == {"Jongens": 1, "Meisjes": 0}
        assert result.state["original_groups"]["Klas A"]["checked_indices"] == [0]


class TestStudentPreferencesSelection:
    """GET /preferences_excel restores the Stap 1 selection saved on download."""

    CANDIDATES = [
        {"key": "s1", "roepnaam": "Anna", "achternaam": "Bos", "groepsnaam": "Groen"},
        {"key": "s2", "roepnaam": "Bram", "achternaam": "Dijk", "groepsnaam": "Groen"},
        {"key": "s3", "roepnaam": "Cas", "achternaam": "El", "groepsnaam": "Blauw"},
    ]

    def _setup(self, client, tmp_path):
        proc_dir = setup_process(client, tmp_path)
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
        html = client.get("/preferences_excel").data.decode("utf-8")
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
        html = client.get("/preferences_excel").data.decode("utf-8")
        assert self._checkbox_state(html) == {"s1": True, "s2": False, "s3": True}
        assert 'value="Daan"' in html
        assert 'value="Jongen" selected' in html
        assert 'value="Blauw" selected' in html


class TestNotTogetherLoadsFromJson:
    """GET /not_together reads student names from voorkeuren.json when it exists."""

    def _mock_groups(self, monkeypatch):
        monkeypatch.setattr(
            wizard_module.datareader,
            "read_groups_excel",
            lambda _: (
                {"klas a": None, "klas b": None},
                {"klas a": "Klas A", "klas b": "Klas B"},
            ),
        )

    def test_renders_student_names_from_voorkeuren_json(
        self, client, tmp_path, monkeypatch
    ):
        """GET /not_together with only voorkeuren.json renders Alice and Bob in the dropdown."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)  # writes Alice + Bob
        self._mock_groups(monkeypatch)

        html = client.get("/not_together").data.decode("utf-8")

        assert "Alice" in html
        assert "Bob" in html

    def test_missing_json_and_xlsx_redirects_with_error(
        self, client, tmp_path, monkeypatch
    ):
        """GET /not_together without any preferences file redirects with an error flash."""
        setup_process(client, tmp_path)
        self._mock_groups(monkeypatch)

        response = client.get("/not_together")

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_excel")
        assert any(cat == "error" for cat, _ in flashes(client))


class TestNotTogetherPage:
    """Tests for POST /not_together error paths."""

    def _mock_file_reads(self, monkeypatch):
        """Patch datareader calls so not_together_page does not need real xlsx files."""
        monkeypatch.setattr(
            wizard_module.datareader,
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
            wizard_module.datareader, "VoorkeurenProcessor", lambda _: mock_proc
        )

    def test_missing_files_flashes_error_and_redirects_to_preferences_excel(
        self, client, tmp_path
    ):
        """not_together_page redirects gracefully when preferences.xlsx is missing."""
        setup_process(client, tmp_path)
        response = client.get("/not_together")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_excel")
        assert any(cat == "error" for cat, _ in flashes(client))

    def test_get_not_together_back_link_points_to_preferences_form_for_form_path(
        self, client, tmp_path, monkeypatch
    ):
        """GET /not_together shows a back link to /preferences_form when input_method=form."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        (proc_dir / "input_method.json").write_text(
            json.dumps({"method": "form"}), encoding="utf-8"
        )
        self._mock_file_reads(monkeypatch)
        response = client.get("/not_together")
        assert response.status_code == 200
        assert b"/preferences_form" in response.data

    def test_post_duplicate_student_flashes_error(self, client, tmp_path, monkeypatch):
        """A rule with the same student listed twice flashes a Dutch parse error."""
        setup_process(client, tmp_path)
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
        assert any(cat == "error" for cat, _ in flashes(client))


class TestUploadPreferencesWritesJson:
    """POST /upload_preferences with a valid Excel also writes voorkeuren.json."""

    _GROUPS = {"blauw": None, "groen": None, "geel": None, "oranje": None}

    def _mock_groups(self, monkeypatch):
        monkeypatch.setattr(
            wizard_module.datareader,
            "read_groups_excel",
            lambda _: (self._GROUPS, {k: k.capitalize() for k in self._GROUPS}),
        )

    def test_valid_upload_writes_voorkeuren_json(self, client, tmp_path, monkeypatch):
        """A successful Excel upload persists voorkeuren.json alongside preferences.xlsx."""
        proc_dir = setup_process(client, tmp_path)
        self._mock_groups(monkeypatch)
        with open("testdata/voorkeuren_klein.xlsx", "rb") as fh:
            raw = fh.read()

        client.post(
            "/upload_preferences",
            data={"preferences": (BytesIO(raw), "voorkeuren.xlsx")},
            content_type="multipart/form-data",
        )

        assert (proc_dir / "voorkeuren.json").exists()
        payload = json.loads((proc_dir / "voorkeuren.json").read_text("utf-8"))
        assert payload.get("source") == "excel"

    def test_valid_upload_still_writes_preferences_xlsx(
        self, client, tmp_path, monkeypatch
    ):
        """preferences.xlsx (for the sociogram) is still written alongside voorkeuren.json."""
        proc_dir = setup_process(client, tmp_path)
        self._mock_groups(monkeypatch)
        with open("testdata/voorkeuren_klein.xlsx", "rb") as fh:
            raw = fh.read()

        client.post(
            "/upload_preferences",
            data={"preferences": (BytesIO(raw), "voorkeuren.xlsx")},
            content_type="multipart/form-data",
        )

        assert (proc_dir / "preferences.xlsx").exists()


class TestStartDistribution:
    """Tests for GET /start_distribution (run lifecycle with a mocked solver)."""

    def _patch_pipeline(self, monkeypatch, *, result=None, exc=None):
        """Run both background threads synchronously with a mocked solver and sociogram.

        Returns the solver mock so a test can inspect the arguments it was called with.
        Both SociogramMaker() and SociogramMaker.from_preference_data() return the same
        mock maker so the mock works for both the Excel and form input paths.
        """
        monkeypatch.setattr(wizard_module, "Thread", immediate_thread)
        solver = MagicMock(side_effect=exc) if exc else MagicMock(return_value=result)
        monkeypatch.setattr(wizard_module, "distribute_students_from_data", solver)
        monkeypatch.setattr(
            wizard_module.datareader,
            "read_groups_excel",
            lambda _: ({"Klas A": None}, {"Klas A": "Klas A"}),
        )
        maker = MagicMock()
        maker.plot_sociogram.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_sociogram_cls = MagicMock()
        mock_sociogram_cls.return_value = maker
        mock_sociogram_cls.from_preference_data.return_value = maker
        monkeypatch.setattr(
            wizard_module.sociogram, "SociogramMaker", mock_sociogram_cls
        )
        fig = MagicMock()
        fig.to_html.return_value = "<div>socio</div>"
        monkeypatch.setattr(
            wizard_module.sociogram, "networkx_to_plotly", lambda *a, **k: fig
        )
        return solver

    def _read_run(self, process_name="testproces"):
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name=process_name
            ).first()
            return proc.run if proc else None

    def test_happy_path_writes_files_and_marks_done(
        self, client, tmp_path, monkeypatch
    ):
        """A successful run writes the artifacts and only then sets status 'done'."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
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
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
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
        """When not_together.json exists it is parsed and passed to the solver."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        (proc_dir / "groups.xlsx").write_bytes(b"dummy")
        (proc_dir / "not_together.json").write_text(
            '[{"group": ["Alice", "Bob"], "Max_aantal_samen": 1}]', encoding="utf-8"
        )
        result = {"download": BytesIO(b"x"), "dataframes": {}}
        solver = self._patch_pipeline(monkeypatch, result=result)

        response = client.get("/start_distribution")
        assert response.status_code == 302
        # distribute_students_from_data(preference_data, target_groups, not_together, ...)
        passed = solver.call_args.args[2]
        assert passed == [{"group": {"Alice", "Bob"}, "Max_aantal_samen": 1}]

    def test_form_path_solver_and_sociogram_succeed_with_only_voorkeuren_json(
        self, client, tmp_path, monkeypatch
    ):
        """Solver and sociogram both complete when only voorkeuren.json exists (form path)."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir, source="form")
        (proc_dir / "groups.xlsx").write_bytes(b"dummy")
        result = {
            "download": BytesIO(b"form-excel"),
            "dataframes": {"Groepsindeling": pd.DataFrame({"A": [1]})},
        }
        self._patch_pipeline(monkeypatch, result=result)

        response = client.get("/start_distribution")
        assert response.status_code == 302
        assert (proc_dir / "results.xlsx").read_bytes() == b"form-excel"
        assert (proc_dir / "sociogram.html").read_text("utf-8") == "<div>socio</div>"
        assert self._read_run().status == "done"


class TestStatus:
    """Tests for GET /status (process-scoped)."""

    def test_no_session_redirects(self, client):
        """Without an active process /status redirects to /processes."""
        response = client.get("/status")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_no_run_returns_unknown_status(self, client, tmp_path):
        """A process without a run row reports status 'unknown' and empty logs."""
        setup_process(client, tmp_path)
        data = client.get("/status").get_json()
        assert data["status_studentdistribution"] == "unknown"
        assert data["logs"] == []

    def test_running_run_returns_status_and_logs(self, client, tmp_path):
        """A running run returns its status and log lines in insertion order."""
        setup_process(client, tmp_path)
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            run = Run(process_id=proc.id, status="running")
            db.session.add(run)
            db.session.add(LogLine(run_id=proc.id, text="Eerste"))
            db.session.add(LogLine(run_id=proc.id, text="Tweede"))
            db.session.commit()
        data = client.get("/status").get_json()
        assert data["status_studentdistribution"] == "running"
        assert data["logs"] == ["Eerste", "Tweede"]

    def test_error_run_includes_message(self, client, tmp_path):
        """An errored run exposes its friendly message for the frontend to flash."""
        setup_process(client, tmp_path)
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            db.session.add(Run(process_id=proc.id, status="error", message="Mislukt"))
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
        setup_process(client, tmp_path)
        response = client.get("/result")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")
        assert any(cat == "error" for cat, _ in flashes(client))

    def test_renders_tables_from_file(self, client, tmp_path):
        """The result page renders the stored HTML tables."""
        proc_dir = setup_process(client, tmp_path)
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
        setup_process(client, tmp_path)
        response = client.get("/sociogram")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_renders_sociogram_file(self, client, tmp_path):
        """A stored sociogram.html is rendered into the page."""
        proc_dir = setup_process(client, tmp_path)
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
        setup_process(client, tmp_path)
        response = client.get("/download")
        assert response.status_code == 200
        # Flash is consumed by base.html during render; verify it appears in the HTML
        assert b"Groepsindeling niet gevonden" in response.data

    def test_existing_file_sends_attachment(self, client, tmp_path):
        """When results.xlsx exists it is sent as an attachment."""
        proc_dir = setup_process(client, tmp_path)
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
        assert any(msg == "Er ging iets mis" for _, msg in flashes(client))


class TestPreferencesForm:
    """Tests for GET/POST /preferences_form."""

    CANDIDATES = [
        {
            "key": "s1",
            "roepnaam": "Anna",
            "achternaam": "Bos",
            "groepsnaam": "Groen",
            "geslacht": "Meisje",
        },
        {
            "key": "s2",
            "roepnaam": "Bram",
            "achternaam": "Dijk",
            "groepsnaam": "Groen",
            "geslacht": "Jongen",
        },
    ]

    def _setup(self, client, tmp_path):
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "relevant_students_and_groups.json").write_text(
            json.dumps({"candidates": self.CANDIDATES, "groups_from": ["Groen"]}),
            encoding="utf-8",
        )
        pd.DataFrame(
            {"Jongens": [1], "Meisjes": [1]}, index=pd.Index(["Klas A"], name="Groepen")
        ).to_excel(proc_dir / "groups.xlsx")
        return proc_dir

    def test_get_returns_200_with_student_names_and_target_groups(
        self, client, tmp_path
    ):
        """GET /preferences_form shows each candidate's name and the target group names."""
        self._setup(client, tmp_path)
        response = client.get("/preferences_form")
        assert response.status_code == 200
        assert b"Anna" in response.data
        assert b"Bram" in response.data
        assert b"Klas A" in response.data

    def test_post_all_checked_writes_voorkeuren_json_and_redirects(
        self, client, tmp_path
    ):
        """POST /preferences_form with all students checked writes voorkeuren.json and
        redirects to not_together."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post(
            "/preferences_form",
            data={"gaat_over": ["s1", "s2"]},
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/not_together")
        payload = json.loads((proc_dir / "voorkeuren.json").read_text("utf-8"))
        assert payload["source"] == "form"
        display_names = set(payload["student_display"].values())
        assert "Anna Bos" in display_names
        assert "Bram Dijk" in display_names

    def test_post_unchecked_student_is_excluded(self, client, tmp_path):
        """POST /preferences_form with only s1 checked excludes Bram from voorkeuren.json."""
        proc_dir = self._setup(client, tmp_path)
        client.post("/preferences_form", data={"gaat_over": ["s1"]})
        payload = json.loads((proc_dir / "voorkeuren.json").read_text("utf-8"))
        display_names = set(payload["student_display"].values())
        assert "Anna Bos" in display_names
        assert "Bram Dijk" not in display_names

    def test_post_with_wish_appears_in_preference_frame(self, client, tmp_path):
        """POST with a 'Graag met' wish is stored in the preference frame."""
        proc_dir = self._setup(client, tmp_path)
        client.post(
            "/preferences_form",
            data={
                "gaat_over": ["s1", "s2"],
                "wens_s1_graag_met_target": ["Bram Dijk"],
                "wens_s1_graag_met_gewicht": ["1"],
            },
        )
        payload = json.loads((proc_dir / "voorkeuren.json").read_text("utf-8"))
        records = payload["preferences"]["records"]
        assert any(
            r["Leerling"] == "annabos"
            and r["TypeWens"] == "Graag met"
            and r["Waarde"] == "bramdijk"
            for r in records
        )

    def test_get_candidates_sorted_by_group(self, client, tmp_path):
        """GET /preferences_form returns candidates sorted by groepsnaam."""
        proc_dir = setup_process(client, tmp_path)
        candidates = [
            {
                "key": "s1",
                "roepnaam": "Zes",
                "achternaam": "Z",
                "groepsnaam": "Zulu",
                "geslacht": "Jongen",
            },
            {
                "key": "s2",
                "roepnaam": "Alfa",
                "achternaam": "A",
                "groepsnaam": "Alpha",
                "geslacht": "Meisje",
            },
        ]
        (proc_dir / "relevant_students_and_groups.json").write_text(
            json.dumps({"candidates": candidates, "groups_from": ["Zulu", "Alpha"]}),
            encoding="utf-8",
        )
        pd.DataFrame(
            {"Jongens": [1, 1], "Meisjes": [0, 1]},
            index=pd.Index(["Klas A", "Klas B"], name="Groepen"),
        ).to_excel(proc_dir / "groups.xlsx")
        html = client.get("/preferences_form").data.decode("utf-8")
        pos_alfa = html.find("Alfa")
        pos_zes = html.find("Zes")
        assert pos_alfa < pos_zes, "Alpha moet vóór Zulu staan"

    def test_post_with_new_student_appears_in_voorkeuren(self, client, tmp_path):
        """POST /preferences_form with a new student (incl. new_key[]) includes them."""
        proc_dir = self._setup(client, tmp_path)
        client.post(
            "/preferences_form",
            data={
                "gaat_over": ["s1", "s2", "new_0"],
                "new_key[]": "new_0",
                "new_voornaam[]": "Emma",
                "new_achternaam[]": "Jansen",
                "new_geslacht[]": "Meisje",
            },
        )
        payload = json.loads((proc_dir / "voorkeuren.json").read_text("utf-8"))
        display_names = set(payload["student_display"].values())
        assert "Emma Jansen" in display_names

    def test_post_min_satisfaction_stored_in_students_info(self, client, tmp_path):
        """POST /preferences_form with min_satisfaction percentage is stored as 0-1 decimal."""
        proc_dir = self._setup(client, tmp_path)
        client.post(
            "/preferences_form",
            data={"gaat_over": ["s1", "s2"], "min_sat_s1": "50"},
        )
        payload = json.loads((proc_dir / "voorkeuren.json").read_text("utf-8"))
        info = payload["students_info"]
        anna_key = next(
            k for k, v in payload["student_display"].items() if v == "Anna Bos"
        )
        assert abs(info[anna_key]["MinimaleTevredenheid"] - 0.5) < 0.001

    def test_post_with_unknown_target_flashes_and_preserves_draft(
        self, client, tmp_path
    ):
        """An invalid preference (unknown target) must not 500: it flashes a friendly
        error, redirects back to the form, preserves the draft, and persists nothing."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post(
            "/preferences_form",
            data={
                "gaat_over": ["s1", "s2"],
                "wens_s1_graag_met_target": ["Spook Persoon"],
                "wens_s1_graag_met_gewicht": ["1"],
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_form")
        # Invalid input is not persisted as the canonical preferences...
        assert not (proc_dir / "voorkeuren.json").exists()
        # ...but the teacher's work is kept as a draft so nothing is lost.
        assert (proc_dir / "preferences_form_state.json").exists()
        # The flash survives the redirect and is shown on the form.
        html = client.get("/preferences_form").data.decode("utf-8")
        assert "flash" in html.lower()

    def test_get_after_post_prefills_wishes_from_state(self, client, tmp_path):
        """GET /preferences_form after a POST prefills the previously saved wish."""
        proc_dir = self._setup(client, tmp_path)
        client.post(
            "/preferences_form",
            data={
                "gaat_over": ["s1", "s2"],
                "wens_s1_graag_met_target": ["Bram Dijk"],
                "wens_s1_graag_met_gewicht": ["2"],
            },
        )
        assert (proc_dir / "preferences_form_state.json").exists()
        html = client.get("/preferences_form").data.decode("utf-8")
        assert "Bram Dijk" in html
