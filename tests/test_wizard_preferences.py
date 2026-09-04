"""Tests for the preference-input wizard routes."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
from io import BytesIO
from unittest.mock import MagicMock

import openpyxl
import pandas as pd

import aliexpress.web.routes.wizard as wizard_module
from aliexpress.web.models import Process
from app import app as flask_app
from tests.helpers import (
    SCHOOL_ID,
    TWO_STUDENTS_GROEN,
    flashes,
    setup_process,
    write_minimal_voorkeuren_json,
)


class TestUploadPreferencesErrors:
    """Tests for friendly preference-upload errors."""

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
        assert 'href="/sociogram"' not in html

    def test_get_shows_sociogram_link_after_preferences_are_saved(
        self, client, tmp_path
    ):
        """The Excel overview links to the sociogram once canonical preferences exist."""
        proc_dir = self._setup(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)

        html = client.get("/preferences_excel").data.decode("utf-8")

        assert 'href="/sociogram"' in html
        assert 'target="_blank"' in html

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


class TestPreferencesFormSociogramLink:  # pylint: disable=too-few-public-methods
    """The form-input overview also exposes the canonical sociogram."""

    def test_get_shows_sociogram_link_after_preferences_are_saved(
        self, client, tmp_path
    ):
        """The form overview links to the sociogram once canonical preferences exist."""
        proc_dir = setup_process(client, tmp_path)
        pd.DataFrame(
            {"Jongens": [1, 1], "Meisjes": [1, 0]},
            index=pd.Index(["Klas A", "Klas B"], name="Groepen"),
        ).to_excel(proc_dir / "groups.xlsx")
        (proc_dir / "roster.json").write_text(
            json.dumps({"participants": TWO_STUDENTS_GROEN}), encoding="utf-8"
        )
        write_minimal_voorkeuren_json(proc_dir)

        html = client.get("/preferences_form").data.decode("utf-8")

        assert 'href="/sociogram"' in html
        assert 'target="_blank"' in html


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

    def test_not_together_continue_text_does_not_claim_to_start_distribution(
        self, client, tmp_path, monkeypatch
    ):
        """The final not-together action accurately describes its next step."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        self._mock_groups(monkeypatch)

        html = client.get("/not_together").data.decode("utf-8")

        assert "Opslaan &amp; door naar indelen" in html
        assert "Opslaan &amp; Indeling starten" not in html

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
