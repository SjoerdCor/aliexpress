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
