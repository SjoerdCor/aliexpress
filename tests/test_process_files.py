"""Tests for web/process_files.py (typed load/save helpers for process artifacts)."""

# pylint: disable=unused-argument  # client fixture used only for its STORAGE_DIR side effect

import os

from aliexpress.solver._balance import BalanceMaxima
from aliexpress.web import process_files
from aliexpress.web.storage import get_process_path
from app import app as flask_app


class TestBalanceMaximaRoundTrip:
    """save_balance_maxima / load_balance_maxima round-trip through balance_limits.json."""

    def test_round_trips_a_mixed_maxima(self, client):
        """Some families capped, some unlimited: the loaded object equals the saved one."""
        school_id, process_id = "test-school", "proc-1"
        with flask_app.app_context():
            os.makedirs(get_process_path(school_id, process_id))
            maxima = BalanceMaxima(
                max_diff_n_students_year=2,
                max_diff_n_students_total=None,
                max_imbalance_boys_girls_year=1,
                max_imbalance_boys_girls_total=None,
                max_clique=5,
                max_clique_sex=None,
            )
            process_files.save_balance_maxima(school_id, process_id, maxima)
            loaded = process_files.load_balance_maxima(school_id, process_id)
        assert loaded == maxima

    def test_round_trips_a_fully_unlimited_maxima(self, client):
        """An all-None BalanceMaxima() round-trips correctly."""
        school_id, process_id = "test-school", "proc-2"
        with flask_app.app_context():
            os.makedirs(get_process_path(school_id, process_id))
            maxima = BalanceMaxima()
            process_files.save_balance_maxima(school_id, process_id, maxima)
            loaded = process_files.load_balance_maxima(school_id, process_id)
        assert loaded == maxima


class TestResetResultFiles:  # pylint: disable=too-few-public-methods
    """reset_result_files removes stale output without touching balance limits."""

    def test_removes_all_stale_outputs_but_keeps_balance_limits(self, client):
        """Every result artifact is removed while the saved limits remain available."""
        school_id, process_id = "test-school", "reset-results"
        result_files = (
            "results.xlsx",
            "result_tables.json",
            "groepsindeling_view.json",
            "sociogram.html",
            "progress.json",
            "interim_result.json",
        )
        with flask_app.app_context():
            process_path = get_process_path(school_id, process_id)
            os.makedirs(process_path)
            for filename in result_files:
                with open(
                    os.path.join(process_path, filename), "w", encoding="utf-8"
                ) as fh:
                    fh.write("stale output")
            process_files.save_balance_maxima(school_id, process_id, BalanceMaxima())

            process_files.reset_result_files(school_id, process_id)

            assert all(
                not os.path.exists(os.path.join(process_path, name))
                for name in result_files
            )
            assert os.path.exists(os.path.join(process_path, "balance_limits.json"))
