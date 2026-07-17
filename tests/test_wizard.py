"""Tests for routes/wizard.py (wizard blueprint)."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern
# pylint: disable=too-many-lines  # large route-test suite; splitting it is separate work

import dataclasses
import json
import re
from dataclasses import asdict
from io import BytesIO
from unittest.mock import MagicMock

import openpyxl
import pandas as pd

import aliexpress.web.routes.wizard as wizard_module
import aliexpress.web.tasks as tasks_module
from aliexpress.errors import ValidationError
from aliexpress.solver._balance import BalanceMaxima
from aliexpress.solver.groepsindeling_view import GroepsindelingView
from aliexpress.web.extensions import db
from aliexpress.web.models import Process, Run
from aliexpress.web.process_files import load_balance_maxima
from app import app as flask_app
from tests.helpers import (
    SCHOOL_ID,
    TWO_STUDENTS_GROEN,
    flashes,
    immediate_thread,
    make_interim_view,
    make_students,
    setup_process,
    write_groups_to_json,
    write_minimal_voorkeuren_json,
)


def _unlimited_maxima_form():
    """Balance-maxima form data with every family ticked Onbeperkt (no numbers required)."""
    return {
        f"maxima_{field.name}_unlimited": "on"
        for field in dataclasses.fields(BalanceMaxima)
    }


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

    def test_post_duplicate_group_names_flashes_error(self, client, tmp_path):
        """POST /groups_to with duplicate group names flashes an error and redirects back."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(proc_dir, {"Klas A": make_students("Jongen")})
        response = client.post(
            "/groups_to",
            data={"group": ["Klas A", "Klas A"]},
        )
        assert response.status_code == 302
        messages = flashes(client)
        assert any(cat == "error" for cat, _ in messages)
        assert any("Klas A" in msg for _, msg in messages)

    def test_post_form_choice_records_method_and_redirects_to_form(
        self, client, tmp_path
    ):
        """POST /groups_to with the 'form' button saves groups.xlsx, records
        input_method=form, and continues to the form preferences page (ADR 0006)."""
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
                "action": "form",
                "group": ["Klas A", "Klas B"],
                "group_students[Klas A]": ["0", "1"],
                "group_students[Klas B]": ["0"],
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_form")
        method = json.loads((proc_dir / "input_method.json").read_text("utf-8"))
        assert method["method"] == "form"

    def test_post_excel_choice_records_method_and_redirects_to_excel(
        self, client, tmp_path
    ):
        """POST /groups_to with the 'excel' button records input_method=excel and continues
        to the Excel preferences page (ADR 0006)."""
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
                "action": "excel",
                "group": ["Klas A", "Klas B"],
                "group_students[Klas A]": ["0", "1"],
                "group_students[Klas B]": ["0"],
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_excel")
        method = json.loads((proc_dir / "input_method.json").read_text("utf-8"))
        assert method["method"] == "excel"

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


class TestGroupsToRedistribute:
    """Tests for /groups_to in redistribute mode (auto-passthrough, no page shown)."""

    def _setup(self, client, tmp_path, groups=("3A", "3B")):
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text('{"mode":"redistribute"}', encoding="utf-8")
        write_groups_to_json(proc_dir, {g: [] for g in groups})
        return proc_dir

    def test_get_auto_redirects_and_writes_groups_xlsx(self, client, tmp_path):
        """GET /groups_to in redistribute mode auto-writes groups.xlsx and redirects."""
        proc_dir = self._setup(client, tmp_path)
        resp = client.get("/groups_to")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/preferences_form")
        df = pd.read_excel(proc_dir / "groups.xlsx", index_col=0)
        assert (df.loc[["3A", "3B"]] == 0).all().all()
        method = json.loads((proc_dir / "input_method.json").read_text("utf-8"))
        assert method["method"] == "form"

    def test_post_also_auto_redirects_to_preferences_form(self, client, tmp_path):
        """POST /groups_to in redistribute mode is the same: transparent passthrough."""
        self._setup(client, tmp_path)
        resp = client.post("/groups_to")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/preferences_form")


class TestPreferencesExcel:
    """GET/POST /preferences_excel: download a roster-prefilled template, then upload it."""

    PARTICIPANTS = TWO_STUDENTS_GROEN

    def _setup(self, client, tmp_path, with_roster=True):
        proc_dir = setup_process(client, tmp_path)
        pd.DataFrame(
            {"Jongens": [1, 1], "Meisjes": [1, 0]},
            index=pd.Index(["Klas A", "Klas B"], name="Groepen"),
        ).to_excel(proc_dir / "groups.xlsx")
        if with_roster:
            (proc_dir / "roster.json").write_text(
                json.dumps({"participants": self.PARTICIPANTS}), encoding="utf-8"
            )
        return proc_dir

    def test_get_redirects_to_roster_when_no_roster_yet(self, client, tmp_path):
        """Without a settled roster the page sends the teacher to 'Wie gaat mee' first."""
        self._setup(client, tmp_path, with_roster=False)
        response = client.get("/preferences_excel")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/roster")

    def test_get_shows_download_without_student_selection(self, client, tmp_path):
        """The page only offers download + upload; there is no per-student selection."""
        self._setup(client, tmp_path)
        html = client.get("/preferences_excel").data.decode("utf-8")
        assert "Download invulformulier" in html
        assert 'name="students"' not in html  # no Stap 1 selection anymore

    def test_post_downloads_excel_prefilled_from_roster(self, client, tmp_path):
        """POST builds the prefilled workbook straight from the roster participants."""
        self._setup(client, tmp_path)
        response = client.post("/preferences_excel")
        assert response.status_code == 200
        assert "voorkeuren.xlsx" in response.headers.get("Content-Disposition", "")
        wb = openpyxl.load_workbook(BytesIO(response.data))
        names = [wb["Sheet1"][f"A{i}"].value for i in range(4, 8)]
        joined = " ".join(n for n in names if n)
        assert "Anna" in joined and "Bram" in joined


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

    def test_post_valid_rules_lands_on_processing_without_starting_a_run(
        self, client, tmp_path, monkeypatch
    ):
        """A valid POST redirects to the idle processing panel; no run is started yet."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        self._mock_file_reads(monkeypatch)

        response = client.post("/not_together", data={"n_rules": "0"})

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processing")
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            assert proc.run is None


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
        monkeypatch.setattr(tasks_module, "distribute_students_from_data", solver)
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
            tasks_module.sociogram, "SociogramMaker", mock_sociogram_cls
        )
        fig = MagicMock()
        fig.to_html.return_value = "<div>socio</div>"
        monkeypatch.setattr(
            tasks_module.sociogram, "networkx_to_plotly", lambda *a, **k: fig
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
            "groepsindeling_view": GroepsindelingView(
                group_order=[], groups=[], balance_rows=[]
            ),
        }
        self._patch_pipeline(monkeypatch, result=result)

        response = client.post("/start_distribution", data=_unlimited_maxima_form())
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processing")
        assert (proc_dir / "results.xlsx").read_bytes() == b"excel-bytes"
        tables = json.loads((proc_dir / "result_tables.json").read_text("utf-8"))
        assert "Groepsindeling" in tables
        view = json.loads((proc_dir / "groepsindeling_view.json").read_text("utf-8"))
        assert view == {"group_order": [], "groups": [], "balance_rows": []}
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

        client.post("/start_distribution", data=_unlimited_maxima_form())
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
        result = {
            "download": BytesIO(b"x"),
            "dataframes": {},
            "groepsindeling_view": GroepsindelingView(
                group_order=[], groups=[], balance_rows=[]
            ),
        }
        solver = self._patch_pipeline(monkeypatch, result=result)

        response = client.post("/start_distribution", data=_unlimited_maxima_form())
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
            "groepsindeling_view": GroepsindelingView(
                group_order=[], groups=[], balance_rows=[]
            ),
        }
        self._patch_pipeline(monkeypatch, result=result)

        response = client.post("/start_distribution", data=_unlimited_maxima_form())
        assert response.status_code == 302
        assert (proc_dir / "results.xlsx").read_bytes() == b"form-excel"
        assert (proc_dir / "sociogram.html").read_text("utf-8") == "<div>socio</div>"
        assert self._read_run().status == "done"

    def test_valid_maxima_are_saved_before_the_solve_starts(
        self, client, tmp_path, monkeypatch
    ):
        """Valid balance-maxima fields are persisted to balance_limits.json and the
        teacher is sent back to /processing to watch the run."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        (proc_dir / "groups.xlsx").write_bytes(b"dummy")
        result = {
            "download": BytesIO(b"x"),
            "dataframes": {},
            "groepsindeling_view": GroepsindelingView(
                group_order=[], groups=[], balance_rows=[]
            ),
        }
        self._patch_pipeline(monkeypatch, result=result)
        form_data = _unlimited_maxima_form()
        del form_data["maxima_max_clique_unlimited"]
        form_data["maxima_max_clique"] = "7"

        response = client.post("/start_distribution", data=form_data)

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processing")
        maxima = load_balance_maxima(SCHOOL_ID, "testproces")
        assert maxima.max_clique == 7
        assert maxima.max_diff_n_students_year is None

    def test_invalid_maxima_flashes_and_does_not_start_a_run(
        self, client, tmp_path, monkeypatch
    ):
        """A malformed balance-maxima field flashes a friendly error and starts no run."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        (proc_dir / "groups.xlsx").write_bytes(b"dummy")
        self._patch_pipeline(monkeypatch)

        response = client.post("/start_distribution", data={})

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processing")
        assert any(cat == "error" for cat, _ in flashes(client))
        assert self._read_run() is None
        assert not (proc_dir / "balance_limits.json").exists()


class TestProcessingIdlePanel:  # pylint: disable=too-few-public-methods  # one test
    """Tests for GET /processing in the idle state (no run started yet)."""

    def test_idle_panel_shows_summary_and_start_button(self, client, tmp_path):
        """The idle panel renders the input summary, the maxima fields and the Start button."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        pd.DataFrame(
            {"Jongens": [1, 1], "Meisjes": [1, 0]},
            index=pd.Index(["klas a", "klas b"], name="Groepen"),
        ).to_excel(proc_dir / "groups.xlsx")

        response = client.get("/processing")

        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert 'name="maxima_max_clique"' in html
        assert "Start verdeling" in html
        assert "leerlingen" in html


class TestStatus:
    """Tests for GET /status (process-scoped)."""

    def test_no_session_redirects(self, client):
        """Without an active process /status redirects to /processes."""
        response = client.get("/status")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_no_run_returns_unknown_status(self, client, tmp_path):
        """A process without a run row reports status 'unknown'."""
        setup_process(client, tmp_path)
        data = client.get("/status").get_json()
        assert data["status_studentdistribution"] == "unknown"

    def test_running_run_reports_sociogram_ready(self, client, tmp_path):
        """A running run reports sociogram_ready based on sociogram.html on disk."""
        proc_dir = setup_process(client, tmp_path)
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            db.session.add(Run(process_id=proc.id, status="running"))
            db.session.commit()

        data = client.get("/status").get_json()
        assert data["status_studentdistribution"] == "running"
        assert data["sociogram_ready"] is False

        (proc_dir / "sociogram.html").write_text("<div>socio</div>", encoding="utf-8")
        data = client.get("/status").get_json()
        assert data["sociogram_ready"] is True

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


class TestInterimResult:
    """Tests for GET /interim_result (process-scoped)."""

    def test_no_session_redirects(self, client):
        """Without an active process /interim_result redirects to /processes."""
        response = client.get("/interim_result")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_no_file_returns_no_content(self, client, tmp_path):
        """Before any interim result was written, the route returns 204."""
        setup_process(client, tmp_path)
        response = client.get("/interim_result")
        assert response.status_code == 204

    def test_renders_view_with_cards(self, client, tmp_path):
        """A stored interim_result.json renders the group cards.

        The "voorlopige indeling / wordt nog verbeterd" caption lives in the
        processing page's <summary> around this partial, not in the partial itself.
        """
        proc_dir = setup_process(client, tmp_path)
        view = make_interim_view()
        (proc_dir / "interim_result.json").write_text(
            json.dumps(asdict(view)), encoding="utf-8"
        )

        response = client.get("/interim_result")
        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert "gi-card" in html
        assert "gi-chip" in html


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


def _make_edexml_reader(df):
    """Return an EdexReader replacement whose get_full_df() returns df."""
    instance = MagicMock()
    instance.get_full_df.return_value = df
    return MagicMock(return_value=instance)


class TestUploadEdexmlMode:
    """Tests for mode-branching in the upload_edexml route."""

    def test_get_forward_mode_shows_jaargroep(self, client, tmp_path):
        """GET /upload_edexml in forward mode shows the jaargroep selector."""
        setup_process(client, tmp_path)
        resp = client.get("/upload_edexml")
        assert resp.status_code == 200
        assert b"jaargroep" in resp.data

    def test_get_redistribute_mode_hides_jaargroep(self, client, tmp_path):
        """GET /upload_edexml in redistribute mode hides the jaargroep selector."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute"}), encoding="utf-8"
        )
        resp = client.get("/upload_edexml")
        assert resp.status_code == 200
        assert b"jaargroep" not in resp.data

    def test_post_redistribute_valid_edexml_redirects_to_select_groups(
        self, client, tmp_path, monkeypatch
    ):
        """POST /upload_edexml in redistribute mode redirects to /select_groups."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute"}), encoding="utf-8"
        )
        fake_df = pd.DataFrame({"groepsnaam": ["3A", "3B"], "jaargroep": [3, 3]})
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={"edexml": (BytesIO(b"anything"), "edex.xml")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/select_groups")

    def test_get_redistribute_and_forward_mode_shows_jaargroep_checkboxes(
        self, client, tmp_path
    ):
        """GET /upload_edexml in redistribute_and_forward mode shows jaargroep checkboxes
        (not the forward-mode dropdown) and the shared "Selecteer leerlingen en groepen"
        button text."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        resp = client.get("/upload_edexml")
        assert resp.status_code == 200
        assert b'name="jaargroep"' not in resp.data
        assert resp.data.count(b'name="jaargroepen"') == 8
        assert "Selecteer leerlingen en groepen".encode() in resp.data

    def test_post_redistribute_and_forward_valid_selection_saves_json_and_redirects_to_roster(
        self, client, tmp_path, monkeypatch
    ):
        """POST /upload_edexml in redistribute_and_forward mode, with jaargroepen ticked,
        determines candidates school-wide for those jaargroepen, leaves groups_to empty
        (destinations are picked later at /select_groups) and redirects to /roster."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        fake_df = pd.DataFrame(
            {
                "groepsnaam": ["5A", "6A", "6B", "7A", "8A"],
                "roepnaam": ["Anna", "Ben", "Carl", "Dana", "Eva"],
                "achternaam": ["A", "B", "C", "D", "E"],
                "key": ["k1", "k2", "k3", "k4", "k5"],
                "geslacht": ["Meisje", "Jongen", "Jongen", "Meisje", "Meisje"],
                "jaargroep": pd.array([5, 6, 6, 7, 8], dtype="Int64"),
            }
        )
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={
                "edexml": (BytesIO(b"anything"), "edex.xml"),
                "jaargroepen": ["5", "6", "7"],
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/roster")
        saved = json.loads(
            (proc_dir / "relevant_students_and_groups.json").read_text("utf-8")
        )
        assert {c["roepnaam"] for c in saved["candidates"]} == {
            "Anna",
            "Ben",
            "Carl",
            "Dana",
        }
        assert saved["groups_to"] == {}
        assert saved["jaargroepen"] == [5, 6, 7]

    def test_post_redistribute_and_forward_without_jaargroep_flashes(
        self, client, tmp_path, monkeypatch
    ):
        """POST /upload_edexml in redistribute_and_forward mode without ticking any
        jaargroep flashes a warning and redirects back to /upload_edexml."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        fake_df = pd.DataFrame(
            {
                "groepsnaam": ["5A"],
                "roepnaam": ["Anna"],
                "achternaam": ["A"],
                "key": ["k1"],
                "geslacht": ["Meisje"],
                "jaargroep": pd.array([5], dtype="Int64"),
            }
        )
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={"edexml": (BytesIO(b"anything"), "edex.xml")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/upload_edexml")
        assert any(cat == "error" for cat, _ in flashes(client))
        assert not (proc_dir / "relevant_students_and_groups.json").exists()

    def test_post_redistribute_and_forward_no_candidates_flashes(
        self, client, tmp_path, monkeypatch
    ):
        """POST /upload_edexml in redistribute_and_forward mode with jaargroepen that match
        no students flashes a warning and redirects back to /upload_edexml."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        fake_df = pd.DataFrame(
            {
                "groepsnaam": ["5A"],
                "roepnaam": ["Anna"],
                "achternaam": ["A"],
                "key": ["k1"],
                "geslacht": ["Meisje"],
                "jaargroep": pd.array([5], dtype="Int64"),
            }
        )
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={
                "edexml": (BytesIO(b"anything"), "edex.xml"),
                "jaargroepen": ["8"],
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/upload_edexml")
        assert any(cat == "error" for cat, _ in flashes(client))
        assert not (proc_dir / "relevant_students_and_groups.json").exists()

    def test_post_redistribute_and_forward_missing_jaargroep_flashes_info(
        self, client, tmp_path, monkeypatch
    ):
        """A student without a jaargroep in the uploaded file is reported (info flash) as
        not taking part, while the rest of the upload succeeds normally."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        fake_df = pd.DataFrame(
            {
                "groepsnaam": ["5A", "5A"],
                "roepnaam": ["Anna", "Ben"],
                "achternaam": ["A", "B"],
                "key": ["k1", "k2"],
                "geslacht": ["Meisje", "Jongen"],
                "jaargroep": pd.array([5, None], dtype="Int64"),
            }
        )
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={
                "edexml": (BytesIO(b"anything"), "edex.xml"),
                "jaargroepen": ["5"],
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/roster")
        assert any(
            cat == "info" and "geen jaargroep" in msg for cat, msg in flashes(client)
        )

    def test_post_redistribute_garbage_edexml_flashes_error(self, client, tmp_path):
        """POST /upload_edexml in redistribute mode with a garbage file flashes an error."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute"}), encoding="utf-8"
        )
        resp = client.post(
            "/upload_edexml",
            data={"edexml": (BytesIO(b"garbage"), "edex.xml")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/upload_edexml")
        assert any(cat == "error" for cat, _ in flashes(client))

    def test_post_redistribute_and_forward_reupload_clears_stale_wizard_state(
        self, client, tmp_path, monkeypatch
    ):
        """Re-uploading EDEXML in redistribute_and_forward mode (e.g. after going back and
        picking different jaargroepen) must wipe stale downstream artifacts from the
        earlier upload, so the roster/preferences pages don't merge old and new
        populations."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        # Simulate leftover state from an earlier upload of jaargroepen 5-6-7.
        (proc_dir / "roster.json").write_text("{}", encoding="utf-8")
        (proc_dir / "voorkeuren.json").write_text("{}", encoding="utf-8")
        (proc_dir / "groups.xlsx").write_text("stale", encoding="utf-8")

        fake_df = pd.DataFrame(
            {
                "groepsnaam": ["3A", "4A", "5A"],
                "roepnaam": ["Anna", "Ben", "Carl"],
                "achternaam": ["A", "B", "C"],
                "key": ["k1", "k2", "k3"],
                "geslacht": ["Meisje", "Jongen", "Jongen"],
                "jaargroep": pd.array([3, 4, 5], dtype="Int64"),
            }
        )
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={
                "edexml": (BytesIO(b"anything"), "edex.xml"),
                "jaargroepen": ["3", "4", "5"],
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/roster")
        assert not (proc_dir / "roster.json").exists()
        assert not (proc_dir / "voorkeuren.json").exists()
        assert not (proc_dir / "groups.xlsx").exists()
        saved = json.loads(
            (proc_dir / "relevant_students_and_groups.json").read_text("utf-8")
        )
        assert {c["roepnaam"] for c in saved["candidates"]} == {"Anna", "Ben", "Carl"}

    def test_post_upload_edexml_removes_stale_downstream_files(
        self, client, tmp_path, monkeypatch
    ):
        """A general check that any new EDEXML upload (forward mode here) removes stale
        wizard artifacts derived from a previous upload."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "roster.json").write_text("{}", encoding="utf-8")
        (proc_dir / "voorkeuren.json").write_text("{}", encoding="utf-8")
        (proc_dir / "groups.xlsx").write_text("stale", encoding="utf-8")

        fake_df = pd.DataFrame({"groepsnaam": ["3A", "3B"], "jaargroep": [3, 3]})
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        monkeypatch.setattr(
            wizard_module.candidatedetermination,
            "handle_edexml_upload",
            lambda df, jaargroep: ([], {}, {}),
        )
        client.post(
            "/upload_edexml",
            data={"edexml": (BytesIO(b"anything"), "edex.xml"), "jaargroep": "4"},
            content_type="multipart/form-data",
        )

        assert not (proc_dir / "roster.json").exists()
        assert not (proc_dir / "voorkeuren.json").exists()
        assert not (proc_dir / "groups.xlsx").exists()
