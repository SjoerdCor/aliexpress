"""Tests for routes/results.py (results blueprint)."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
import re
from dataclasses import asdict

import pandas as pd

import aliexpress.web.routes.results as results_module
from aliexpress.data.preferences_form import Preference, PreferenceKind, StudentEntry
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


def _write_result_view(proc_dir):
    """Persist the shared minimal structured result fixture for result-route tests."""
    (proc_dir / "groepsindeling_view.json").write_text(
        json.dumps(asdict(make_interim_view())), encoding="utf-8"
    )


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
        assert "Groepsindeling berekenen →" in html
        assert "leerlingen" in html

    def test_difference_edit_query_opens_balance_fields(self, client, tmp_path):
        """The adjustment link opens the existing balance-limit disclosure."""
        proc_dir = _setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        write_minimal_groups_xlsx(proc_dir)

        html = client.get("/processing?edit=differences").data.decode("utf-8")

        details_tag = re.search(r'<details class="instructions-box"[^>]*>', html)
        assert details_tag is not None
        assert " open" in details_tag.group()


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
        assert "Groepsindeling berekenen →" in html
        assert "Groepsindeling opnieuw berekenen →" not in html
        assert "Een nieuwe berekening vervangt de huidige groepsindeling." in html
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

    def test_pending_run_shows_progress_view_without_ready_summary(
        self, client, tmp_path
    ):
        """A pending run is active already and must not show the ready summary or Start button."""
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
        assert "Groepsindeling berekenen" in html
        assert 'id="input-overview"' not in html
        assert "Groepsindeling berekenen →" not in html

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
        assert re.search(r'name="maxima_max_clique_sex_unlimited"[^>]*\schecked', html)
        assert "Mislukt" in html
        assert 'class="flash-message error"' in html
        assert "Dit kun je aanpassen" not in html
        assert 'class="calculation-error"' not in html
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

    def test_status_reports_run_and_progress_without_sociogram_state(
        self, client, tmp_path
    ):
        """Status contains solver state and progress, not visualisation state."""
        proc_dir = _setup_process(client, tmp_path)
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            db.session.add(Run(process_id=proc.id, status="running"))
            db.session.commit()

        data = client.get("/status").get_json()
        assert data["status_studentdistribution"] == "running"

        (proc_dir / "progress.json").write_text(
            json.dumps({"steps": {"floor": "busy"}}), encoding="utf-8"
        )
        data = client.get("/status").get_json()
        assert data["status_studentdistribution"] == "running"
        assert data["steps"] == {"floor": "busy"}

    def test_error_run_includes_message(self, client, tmp_path):
        """An errored run exposes its friendly message for the processing page."""
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


class TestProcessingSummary:  # pylint: disable=too-few-public-methods
    """The ready summary is complete, semantic, and does not start a run."""

    def test_ready_get_is_read_only_and_renders_server_summary_once(
        self, client, tmp_path
    ):
        """A ready GET renders the full summary once without starting a run."""
        proc_dir = _setup_process(client, tmp_path)
        students = [
            StudentEntry(
                "Alexandra van de Water",
                "Meisje",
                "Oude groep met een bijzonder lange naam",
                None,
                preferences=[
                    Preference("Bram van den Berg", 1.0, PreferenceKind.TOGETHER)
                ],
            ),
            StudentEntry(
                "Bram van den Berg",
                "Jongen",
                "Oude groep met een bijzonder lange naam",
                None,
                preferences=[
                    Preference("Alexandra van de Water", 1.0, PreferenceKind.APART)
                ],
            ),
            StudentEntry(
                "Cato de Vries",
                "Meisje",
                "Andere lange huidige groep",
                0.5,
                excluded_groups=["Een nieuwe groep met een lange naam"],
            ),
        ]
        write_minimal_voorkeuren_json(
            proc_dir,
            students=students,
            all_to_groups=[
                "eennieuwegroepmeteenlangenaam",
                "nogenieuwegroepmeteenlangenaam",
            ],
        )
        write_minimal_groups_xlsx(proc_dir)

        html = client.get("/processing").get_data(as_text=True)

        assert "Groepsindeling berekenen" in html
        assert "Oude groep met een bijzonder lange naam" in html
        pd.DataFrame(
            {"Jongens": [1, 1], "Meisjes": [1, 0]},
            index=pd.Index(
                [
                    "Een nieuwe groep met een lange naam",
                    "Nog een nieuwe groep met een lange naam",
                ],
                name="Groepen",
            ),
        ).to_excel(proc_dir / "groups.xlsx")
        html = client.get("/processing").get_data(as_text=True)
        assert "Groepen in deze indeling (2)" in html
        assert "Een nieuwe groep met een lange naam" in html
        assert "Nog een nieuwe groep met een lange naam" in html
        assert "2 leerlingen met één of meer voorkeuren" in html
        assert "Huidige jaarlaag" not in html
        assert "voor 2 van 3" not in html
        assert html.count('id="input-overview"') == 1
        assert not (proc_dir / "progress.json").exists()
        with flask_app.app_context():
            proc = Process.by_name(SCHOOL_ID, "testproces")
            assert proc.run is None


class TestResultPage:
    """Tests for GET /result (process-scoped)."""

    def test_no_session_redirects(self, client):
        """Without an active process /result redirects to /processes."""
        response = client.get("/result")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_missing_view_flashes_and_redirects(self, client, tmp_path):
        """Visiting /result before the structured view exists flashes an error and redirects."""
        _setup_process(client, tmp_path)
        response = client.get("/result")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")
        assert any(cat == "error" for cat, _ in flashes(client))

    def test_result_page_shows_sociogram_link(self, client, tmp_path):
        """The completed-result page keeps the direct sociogram link."""
        proc_dir = _setup_process(client, tmp_path)
        _write_result_view(proc_dir)

        html = client.get("/result").data.decode("utf-8")

        assert 'href="/sociogram"' in html
        assert 'target="_blank"' in html

    def test_adjustment_links_open_existing_input_steps(self, client, tmp_path):
        """The result page links to existing input steps without a new route."""
        proc_dir = _setup_process(client, tmp_path)
        _write_result_view(proc_dir)

        html = client.get("/result").data.decode("utf-8")

        assert "← Nog niet helemaal... opnieuw invoeren" in html
        assert re.search(
            r'href="/preferences_form"[^>]*>Voorkeuren aanpassen</a>',
            html,
        )
        assert re.search(
            r'href="/not_together"[^>]*>Leerlingen spreiden aanpassen</a>', html
        )
        assert re.search(
            r'href="/processing\?edit=differences"[^>]*>'
            r"Ruimte voor verschillen tussen groepen aanpassen</a>",
            html,
        )
        adjustment = html.split("Nog niet helemaal... opnieuw invoeren", 1)[1]
        assert "grotere verschillen" in adjustment
        assert "toegestane verschillen tussen groepen kleiner" in adjustment
        assert "Wil je juist gelijkere groepen" not in adjustment

    def test_structured_view_renders_native_result_analyses(self, client, tmp_path):
        """A structured result exposes sorted native student and origin analyses."""
        proc_dir = _setup_process(client, tmp_path)
        view = {
            "group_order": ["Groep A", "Groep B"],
            "groups": [
                {
                    "name": "Groep A",
                    "total": 3,
                    "boys_total": 2,
                    "girls_total": 1,
                    "year_sections": [
                        {
                            "year": 6,
                            "label": "Jaarlaag 6",
                            "size": 3,
                            "boys": {
                                "sex": "Jongen",
                                "new_count": 2,
                                "students": [
                                    {
                                        "chip_name": "Bob",
                                        "full_name": "Bob Lange Naam",
                                        "origin_abbrev": "Sta",
                                        "origin_full": "Stam 1",
                                        "year_group": 6,
                                        "satisfaction": 0.5,
                                        "preferences": [
                                            {
                                                "kind": "graag_met",
                                                "target": "Cato",
                                                "fulfilled": True,
                                                "target_is_group": False,
                                                "weight": 1.0,
                                            }
                                        ],
                                        "not_in": [],
                                        "min_satisfaction": None,
                                    },
                                    {
                                        "chip_name": "Daan",
                                        "full_name": "Daan",
                                        "origin_abbrev": "Sta",
                                        "origin_full": "Stam 2",
                                        "year_group": 6,
                                        "satisfaction": 1.0,
                                        "preferences": [
                                            {
                                                "kind": "graag_met",
                                                "target": "Groep B",
                                                "fulfilled": True,
                                                "target_is_group": True,
                                                "weight": 5.0,
                                            }
                                        ],
                                        "not_in": [],
                                        "min_satisfaction": "full",
                                    },
                                ],
                            },
                            "girls": {
                                "sex": "Meisje",
                                "new_count": 1,
                                "students": [
                                    {
                                        "chip_name": "Alice",
                                        "full_name": "Alice",
                                        "origin_abbrev": "Sta",
                                        "origin_full": "Stam 1",
                                        "year_group": 6,
                                        "satisfaction": 0.25,
                                        "preferences": [
                                            {
                                                "kind": "graag_met",
                                                "target": "Bob",
                                                "fulfilled": True,
                                                "target_is_group": False,
                                                "weight": 0.5,
                                            },
                                            {
                                                "kind": "liever_niet_met",
                                                "target": "Groep B",
                                                "fulfilled": False,
                                                "target_is_group": True,
                                                "weight": 2.0,
                                            },
                                        ],
                                        "not_in": [],
                                        "min_satisfaction": "partial",
                                    }
                                ],
                            },
                        }
                    ],
                },
                {
                    "name": "Groep B",
                    "total": 2,
                    "boys_total": 1,
                    "girls_total": 1,
                    "year_sections": [
                        {
                            "year": 6,
                            "label": "Jaarlaag 6",
                            "size": 2,
                            "boys": {
                                "sex": "Jongen",
                                "new_count": 1,
                                "students": [
                                    {
                                        "chip_name": "Geen",
                                        "full_name": "Geen Voorkeur",
                                        "origin_abbrev": "Sta",
                                        "origin_full": "Stam 2",
                                        "year_group": 6,
                                        "satisfaction": None,
                                        "preferences": [],
                                        "not_in": ["Groep A"],
                                        "min_satisfaction": "partial",
                                    }
                                ],
                            },
                            "girls": {
                                "sex": "Meisje",
                                "new_count": 1,
                                "students": [
                                    {
                                        "chip_name": "Cato",
                                        "full_name": "Cato",
                                        "origin_abbrev": "Sta",
                                        "origin_full": "Stam 1",
                                        "year_group": 6,
                                        "satisfaction": 0.0,
                                        "preferences": [
                                            {
                                                "kind": "liever_niet_met",
                                                "target": "Alice",
                                                "fulfilled": True,
                                                "target_is_group": False,
                                                "weight": 1.0,
                                            },
                                            {
                                                "kind": "liever_niet_met",
                                                "target": "Bob",
                                                "fulfilled": False,
                                                "target_is_group": False,
                                                "weight": 1.0,
                                            },
                                        ],
                                        "not_in": [],
                                        "min_satisfaction": None,
                                    }
                                ],
                            },
                        }
                    ],
                },
            ],
            "balance_rows": [
                {
                    "label": "Totaal",
                    "is_total": True,
                    "per_group": {
                        "Groep A": [3, 2, 1],
                        "Groep B": [2, 1, 1],
                    },
                    "size_diff": 1,
                    "sex_imbalance": 1,
                },
                {
                    "label": "Jaarlaag 6",
                    "is_total": False,
                    "per_group": {
                        "Groep A": [3, 2, 1],
                        "Groep B": [2, 1, 1],
                    },
                    "size_diff": 1,
                    "sex_imbalance": 1,
                },
            ],
        }
        (proc_dir / "groepsindeling_view.json").write_text(
            json.dumps(view), encoding="utf-8"
        )

        html = client.get("/result").data.decode("utf-8")

        assert "Je groepsindeling is klaar!" in html
        assert "Tevredenheid en voorkeuren per leerling" in html
        assert "Waar komen de leerlingen vandaan?" in html
        assert "Maximaal 2 leerlingen uit dezelfde huidige groep" in html
        assert "Hele groep of jaarlaag" in html
        assert "Verschil tussen grootste en kleinste groep" in html
        assert "Grootste verschil tussen jongens en meisjes binnen een groep" in html
        assert html.count("Jaarlaag 6") >= 4
        assert "1 leerling" in html
        assert "1 leerlingen" not in html
        assert "Het verschil in groepsgrootte is" not in html
        student_analysis = html.split("Tevredenheid en voorkeuren per leerling", 1)[1]
        assert student_analysis.index("Daan") < student_analysis.index("Bob Lange Naam")
        assert "2 van 2" not in html
        assert "1 van 2 voorkeuren" not in html
        assert "Geen voorkeuren ingevuld" in html
        student_rows = student_analysis.split("</details>", 1)[0]
        assert ">Gehonoreerd<" not in student_rows
        assert ">Niet gehonoreerd<" not in student_rows
        assert "Graag bij Bob" in student_analysis
        assert "Liever niet in Groep B" in student_analysis
        assert student_analysis.index("Graag bij Bob") < student_analysis.index(
            "Liever niet in Groep B"
        )
        assert student_analysis.index("Liever niet bij Alice") < student_analysis.index(
            "Liever niet bij Bob"
        )
        assert "25%" in student_analysis
        assert "25.00%" not in student_analysis
        assert "♥" in student_analysis
        assert "↑" in student_analysis
        assert "~" in student_analysis
        assert "Niet in" not in student_rows
        assert "Extra zekerheid:" not in student_rows
        assert "De verschillen zijn te groot." in html
        assert html.count('href="/processing?edit=differences"') >= 2


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

    def test_missing_preferences_flashes_an_error_on_sociogram_page(
        self, client, tmp_path
    ):
        """Missing canonical preferences use the application's normal flash convention."""
        _setup_process(client, tmp_path)
        response = client.get("/sociogram")
        assert response.status_code == 200
        assert b"Sociogram niet beschikbaar" in response.data
        assert b"geldige voorkeuren ontbreken" in response.data

    def test_unreadable_preferences_do_not_change_run_status(self, client, tmp_path):
        """A broken preferences file is isolated from an active solver run."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "voorkeuren.json").write_text("geen json", encoding="utf-8")
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            db.session.add(Run(process_id=proc.id, status="running"))
            db.session.commit()

        response = client.get("/sociogram")

        assert response.status_code == 200
        assert b"Sociogram niet beschikbaar" in response.data
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            assert proc.run.status == "running"

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

    def test_missing_file_redirects_to_result_page_with_flash(self, client, tmp_path):
        """A missing workbook returns to the normal result page with a flash."""
        proc_dir = _setup_process(client, tmp_path)
        _write_result_view(proc_dir)
        response = client.get("/download")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/result")
        result_response = client.get(response.headers["Location"])
        assert result_response.status_code == 200
        assert b"Groepsindeling niet gevonden" in result_response.data

    def test_existing_file_sends_attachment(self, client, tmp_path):
        """When results.xlsx exists it is sent as an attachment."""
        proc_dir = _setup_process(client, tmp_path)
        (proc_dir / "results.xlsx").write_bytes(b"dummy excel content")
        response = client.get("/download")
        assert response.status_code == 200
        assert "attachment" in response.headers.get("Content-Disposition", "")
