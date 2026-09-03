"""Tests for routes/results.py (results blueprint)."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
import re
from dataclasses import asdict

import aliexpress.web.routes.results as results_module
from aliexpress.solver._balance import BalanceMaxima
from aliexpress.web.extensions import db
from aliexpress.web.models import Process, Run
from aliexpress.web.process_files import load_balance_maxima, save_balance_maxima
from app import app as flask_app
from tests.helpers import (
    flashes,
    make_interim_view,
    write_minimal_groups_xlsx,
    write_minimal_voorkeuren_json,
)

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


class TestProcessingIdlePanel:  # pylint: disable=too-few-public-methods  # one test
    """Tests for GET /processing in the idle state (no run started yet)."""

    def test_idle_panel_shows_summary_and_start_button(self, client, tmp_path):
        """The idle panel renders the input summary, the maxima fields and the Start button."""
        proc_dir = _setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        write_minimal_groups_xlsx(proc_dir)

        response = client.get("/processing")

        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert 'name="maxima_max_clique"' in html
        assert "Start verdeling" in html
        assert "leerlingen" in html


class TestProcessingRunStates:
    """Tests for the processing page while a run is active."""

    def test_done_run_opens_safe_recalculation_form_without_mutating_state(
        self, client, tmp_path
    ):
        """A completed run can be revisited as an idle form without side effects."""
        proc_dir = _setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        write_minimal_groups_xlsx(proc_dir)
        saved_maxima = BalanceMaxima(
            max_diff_n_students_year=6,
            max_diff_n_students_total=8,
            max_imbalance_boys_girls_year=5,
            max_imbalance_boys_girls_total=7,
            max_clique=4,
            max_clique_sex=3,
        )
        save_balance_maxima(SCHOOL_ID, "testproces", saved_maxima)
        result_files = {
            "results.xlsx": b"existing workbook",
            "result_tables.json": b'{"existing": "tables"}',
            "groepsindeling_view.json": b'{"existing": "view"}',
        }
        for filename, contents in result_files.items():
            (proc_dir / filename).write_bytes(contents)
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            db.session.add(Run(process_id=proc.id, status="done"))
            db.session.commit()

        response = client.get("/processing")

        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert re.search(r'name="maxima_max_diff_n_students_year"[^>]*value="6"', html)
        assert re.search(r'name="maxima_max_clique"[^>]*value="4"', html)
        assert "Start nieuwe indeling" in html
        assert "Start verdeling" not in html
        assert "Een nieuwe indeling vervangt het huidige resultaat." in html
        assert re.search(
            r'href="/download"[^>]*>Download huidige groepsindeling</a>', html
        )
        details_tag = re.search(
            r'<details class="instructions-box"[^>]*>', html
        ).group()
        assert " open" not in details_tag

        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            assert proc.run.status == "done"
        assert load_balance_maxima(SCHOOL_ID, "testproces") == saved_maxima
        for filename, contents in result_files.items():
            assert (proc_dir / filename).read_bytes() == contents

    def test_done_run_in_watch_mode_redirects_to_result(self, client, tmp_path):
        """The explicit processing watch mode follows a completed run to its result."""
        _setup_process(client, tmp_path)
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            db.session.add(Run(process_id=proc.id, status="done"))
            db.session.commit()

        response = client.get("/processing?watch=1")

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/result")

    def test_pending_run_shows_progress_view_with_server_summary(
        self, client, tmp_path
    ):
        """A pending run is active already and must not show the Start button."""
        proc_dir = _setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        write_minimal_groups_xlsx(proc_dir)
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            db.session.add(Run(process_id=proc.id, status="pending"))
            db.session.commit()

        response = client.get("/processing")

        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert "Groepsindeling aan het uitrekenen" in html
        assert (
            'id="input-overview" class="input-overview input-overview--visible"' in html
        )
        assert "2 leerlingen" in html
        assert "Start verdeling" not in html

    def test_error_run_reuses_saved_balance_maxima(self, client, tmp_path):
        """An error page shows the limits chosen for the failed attempt."""
        proc_dir = _setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        write_minimal_groups_xlsx(proc_dir)
        save_balance_maxima(
            SCHOOL_ID,
            "testproces",
            BalanceMaxima(max_diff_n_students_year=6, max_clique=7),
        )
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            db.session.add(Run(process_id=proc.id, status="error", message="Mislukt"))
            db.session.commit()

        response = client.get("/processing")

        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert 'name="maxima_max_diff_n_students_year"' in html
        assert 'name="maxima_max_clique"' in html
        assert 'value="6"' in html
        assert 'value="7"' in html
        assert 'value="None"' not in html
        assert re.search(r'name="maxima_max_clique_sex_unlimited"\s+checked', html)
        details_tag = re.search(
            r'<details class="instructions-box"[^>]*>', html
        ).group()
        assert " open" in details_tag


class TestStatus:
    """Tests for GET /status (process-scoped)."""

    def test_no_session_redirects(self, client):
        """Without an active process /status redirects to /processes."""
        response = client.get("/status")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_no_run_returns_unknown_status(self, client, tmp_path):
        """A process without a run row reports status 'unknown'."""
        _setup_process(client, tmp_path)
        data = client.get("/status").get_json()
        assert data["status_studentdistribution"] == "unknown"

    def test_running_run_reports_sociogram_ready(self, client, tmp_path):
        """A running run reports sociogram_ready based on sociogram.html on disk."""
        proc_dir = _setup_process(client, tmp_path)
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
        _setup_process(client, tmp_path)
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
        _setup_process(client, tmp_path)
        response = client.get("/result")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")
        assert any(cat == "error" for cat, _ in flashes(client))

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

    def test_restart_link_opens_plain_processing_form(self, client, tmp_path):
        """The existing retry link opens editable processing without watch mode."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "result_tables.json").write_text("{}", encoding="utf-8")

        html = client.get("/result").data.decode("utf-8")

        assert "\u2190 Nog niet helemaal... opnieuw invoeren" in html
        assert re.search(
            r'href="/processing"[^>]*>\u2190 Nog niet helemaal\.\.\. opnieuw invoeren</a>',
            html,
        )
        assert "/processing?watch=" not in html


class TestInterimResult:
    """Tests for GET /interim_result (process-scoped)."""

    def test_no_session_redirects(self, client):
        """Without an active process /interim_result redirects to /processes."""
        response = client.get("/interim_result")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_no_file_returns_no_content(self, client, tmp_path):
        """Before any interim result was written, the route returns 204."""
        _setup_process(client, tmp_path)
        response = client.get("/interim_result")
        assert response.status_code == 204

    def test_renders_view_with_cards(self, client, tmp_path):
        """A stored interim_result.json renders the group cards.

        The "voorlopige indeling / wordt nog verbeterd" caption lives in the
        processing page's <summary> around this partial, not in the partial itself.
        """
        proc_dir = _setup_process(client, tmp_path)
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

    def test_missing_preferences_flashes_and_redirects(self, client, tmp_path):
        """Visiting /sociogram without canonical preferences flashes an error."""
        _setup_process(client, tmp_path)
        response = client.get("/sociogram")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_renders_sociogram_from_preference_data(self, client, tmp_path):
        """The route builds visible nodes and arrows from voorkeuren.json."""
        proc_dir = _setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        response = client.get("/sociogram")
        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert '"label": "Alice"' in html
        assert '"label": "Bob"' in html
        assert '"source": "alice"' in html
        assert '"target": "bob"' in html
        assert '"weight": 1.0' in html
        assert "cytoscape-3.34.0.min.js" in html


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
        assert any(msg == "Er ging iets mis" for _, msg in flashes(client))
