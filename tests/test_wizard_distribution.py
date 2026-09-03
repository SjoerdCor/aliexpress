"""Tests for the /start_distribution wizard route."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern
# pylint: disable=duplicate-code  # route setup repeats intentional integration scenarios

import dataclasses
import json
import re
from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd
import pytest

import aliexpress.web.routes.wizard as wizard_module
import aliexpress.web.tasks as tasks_module
from aliexpress.errors import FeasibilityError, ValidationError
from aliexpress.solver._balance import BalanceMaxima
from aliexpress.solver.groepsindeling_view import GroepsindelingView
from aliexpress.web.extensions import db
from aliexpress.web.models import Process, Run
from aliexpress.web.process_files import load_balance_maxima, save_balance_maxima
from app import app as flask_app
from tests.helpers import (
    SCHOOL_ID,
    flashes,
    immediate_thread,
    setup_process,
    write_minimal_groups_xlsx,
    write_minimal_voorkeuren_json,
)


def _unlimited_maxima_form():
    """Balance-maxima form data with every family ticked Onbeperkt (no numbers required)."""
    return {
        f"maxima_{field.name}_unlimited": "on"
        for field in dataclasses.fields(BalanceMaxima)
    }


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
        assert response.headers["Location"].endswith("/processing?watch=1")
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

    def test_detailed_conflict_keeps_multiline_message_in_processing_flow(
        self, client, tmp_path, monkeypatch
    ):
        """A detailed diagnosis follows the existing status and flash-message path."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        write_minimal_groups_xlsx(proc_dir)
        exc = FeasibilityError(
            "infeasible_preferences",
            context={
                "case": "detailed",
                "conflict": {
                    "conditions": [
                        {
                            "type": "minimum_satisfaction",
                            "student": "Piet",
                            "floor": 1.0,
                            "preferences": [
                                {
                                    "kind": "Graag met",
                                    "target": "Sam",
                                    "weight": 1.0,
                                }
                            ],
                        },
                        {
                            "type": "forbidden_group",
                            "student": "Piet",
                            "group": "Blauw",
                        },
                    ]
                },
            },
        )
        self._patch_pipeline(monkeypatch, exc=exc)

        client.post("/start_distribution", data=_unlimited_maxima_form())
        status = client.get("/status").get_json()
        assert status["status_studentdistribution"] == "error"
        assert "extra zekerheid" in status["message"]
        assert "\n" in status["message"]

        client.post("/handle-error", json={"message": status["message"]})
        html = client.get("/processing").data.decode("utf-8")
        assert "Piet" in html
        assert "Graag met Sam" in html

    def test_balance_cap_error_returns_to_idle_with_message_and_saved_limits(
        self, client, tmp_path, monkeypatch
    ):
        """The processing error flow shows the cap tip and keeps entered limits."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        write_minimal_groups_xlsx(proc_dir)
        exc = FeasibilityError(
            "balance_caps_too_tight",
            context={
                "suggestion": {
                    "clique": {"current": 1, "suggested": 2},
                }
            },
        )
        self._patch_pipeline(monkeypatch, exc=exc)
        form_data = _unlimited_maxima_form()
        del form_data["maxima_max_clique_unlimited"]
        form_data["maxima_max_clique"] = "1"

        response = client.post("/start_distribution", data=form_data)
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processing?watch=1")

        status = client.get("/status").get_json()
        assert status["status_studentdistribution"] == "error"
        assert "Zelfde stamgroep totaal" in status["message"]
        assert "van 1 naar 2 (+1)" in status["message"]

        client.post("/handle-error", json={"message": status["message"]})
        processing = client.get("/processing")
        assert processing.status_code == 200
        html = processing.data.decode("utf-8")
        assert "Met deze grenzen is geen geldige indeling mogelijk." in html
        assert "Start verdeling" in html
        assert re.search(r'name="maxima_max_clique"[^>]*value="1"', html)

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
        assert response.headers["Location"].endswith("/processing?watch=1")
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

    @pytest.mark.parametrize("status", ["pending", "running"])
    def test_active_run_rejects_second_start_without_changing_files(
        self, client, tmp_path, monkeypatch, status
    ):
        """A repeated valid POST cannot replace an active run or its artifacts."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        write_minimal_groups_xlsx(proc_dir)
        save_balance_maxima(
            SCHOOL_ID,
            "testproces",
            BalanceMaxima(max_clique=4),
        )
        old_outputs = {
            "results.xlsx": b"old workbook",
            "result_tables.json": b'{"old": true}',
            "groepsindeling_view.json": b'{"old": true}',
        }
        for name, content in old_outputs.items():
            (proc_dir / name).write_bytes(content)
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            db.session.add(Run(process_id=proc.id, status=status, message="behouden"))
            db.session.commit()
            created_at = proc.run.created_at

        thread_factory = MagicMock()
        monkeypatch.setattr(wizard_module, "Thread", thread_factory)
        form_data = _unlimited_maxima_form()
        del form_data["maxima_max_clique_unlimited"]
        form_data["maxima_max_clique"] = "7"

        response = client.post("/start_distribution", data=form_data)

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processing?watch=1")
        assert flashes(client) == [("info", "De groepsindeling wordt al berekend.")]
        thread_factory.assert_not_called()
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            assert proc.run.status == status
            assert proc.run.message == "behouden"
            assert proc.run.created_at == created_at
        assert load_balance_maxima(SCHOOL_ID, "testproces").max_clique == 4
        for name, content in old_outputs.items():
            assert (proc_dir / name).read_bytes() == content

    def test_accepted_restart_removes_every_old_output_but_keeps_new_limits(
        self, client, tmp_path, monkeypatch
    ):
        """A claimed restart clears stale output before its workers are spawned."""
        proc_dir = setup_process(client, tmp_path)
        write_minimal_voorkeuren_json(proc_dir)
        write_minimal_groups_xlsx(proc_dir)
        old_outputs = """results.xlsx result_tables.json groepsindeling_view.json
        sociogram.html progress.json interim_result.json""".split()
        for name in old_outputs:
            (proc_dir / name).write_bytes(b"old output")
        with flask_app.app_context():
            proc = Process.query.filter_by(
                school_id=SCHOOL_ID, name="testproces"
            ).first()
            db.session.add(Run(process_id=proc.id, status="done"))
            db.session.commit()

        thread_factory = MagicMock()
        monkeypatch.setattr(wizard_module, "Thread", thread_factory)
        form_data = _unlimited_maxima_form()
        del form_data["maxima_max_clique_unlimited"]
        form_data["maxima_max_clique"] = "7"

        response = client.post("/start_distribution", data=form_data)

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processing?watch=1")
        assert thread_factory.call_count == 2
        assert all(not (proc_dir / name).exists() for name in old_outputs)
        assert load_balance_maxima(SCHOOL_ID, "testproces").max_clique == 7
        assert self._read_run().status == "pending"
