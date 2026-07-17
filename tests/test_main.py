"""Unit tests for main.py orchestration helpers."""

import logging

from aliexpress.main import _log_initial_state, build_input_summary


def test_log_initial_state_reports_stamgroep_as_entered(monkeypatch):
    """The initial-state log lines show Stamgroep names as entered, not matching keys."""
    groups_to = {"blauw": {"Jongens": 1, "Meisjes": 1}}
    students_info = {
        "jan": {"Jongen/meisje": "Jongen", "Stamgroep": "klasa"},
        "eva": {"Jongen/meisje": "Meisje", "Stamgroep": "klasa"},
        "tom": {"Jongen/meisje": "Jongen", "Stamgroep": "annes"},
    }
    stamgroep_display = {"klasa": "Klas A", "annes": "Anne's groep"}

    logged: list[str] = []
    monkeypatch.setattr(
        logging.getLogger("aliexpress.main"),
        "info",
        lambda msg, *args: logged.append(msg % args),
    )

    _log_initial_state(groups_to, students_info, stamgroep_display)

    full_log = "\n".join(logged)
    assert "Klas A" in full_log
    assert "Anne's groep" in full_log
    # The internal matching keys must not leak into the log messages.
    assert "klasa" not in full_log


def test_build_input_summary_counts_display_groups_not_matching_keys():
    """The input overview counts display Stamgroepen, not the internal matching keys."""
    groups_to = {
        "blauw": {"Jongens": 1, "Meisjes": 1},
        "rood": {"Jongens": 1, "Meisjes": 1},
    }
    students_info = {
        "jan": {"Jongen/meisje": "Jongen", "Stamgroep": "klasa", "Jaarlaag": 6},
        "eva": {"Jongen/meisje": "Meisje", "Stamgroep": "klasa2", "Jaarlaag": 6},
        "tom": {"Jongen/meisje": "Jongen", "Stamgroep": "annes", "Jaarlaag": 7},
    }
    # jan and eva sit in two *different* matching keys ("klasa", "klasa2") that map to
    # the *same* display name "Klas A". Counting matching keys would give three source
    # groups; grouping by display name collapses them to "Klas A" with 2 students.
    stamgroep_display = {"klasa": "Klas A", "klasa2": "Klas A", "annes": "Anne's groep"}

    summary = build_input_summary(groups_to, students_info, stamgroep_display)

    assert summary.n_students == 3
    assert summary.n_boys == 2
    assert summary.n_girls == 1
    assert summary.source_groups == {"Klas A": 2, "Anne's groep": 1}
    assert summary.n_target_groups == 2
    assert summary.years == [6, 7]
