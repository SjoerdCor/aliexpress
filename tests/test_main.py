"""Unit tests for main.py orchestration helpers."""

from aliexpress.main import _log_initial_state


def test_log_initial_state_reports_stamgroep_as_entered():
    """The processing-page messages show Stamgroep names as entered, not matching keys."""
    messages = []
    groups_to = {"blauw": {"Jongens": 1, "Meisjes": 1}}
    students_info = {
        "jan": {"Jongen/meisje": "Jongen", "Stamgroep": "klasa"},
        "eva": {"Jongen/meisje": "Meisje", "Stamgroep": "klasa"},
        "tom": {"Jongen/meisje": "Jongen", "Stamgroep": "annes"},
    }
    stamgroep_display = {"klasa": "Klas A", "annes": "Anne's groep"}

    _log_initial_state(groups_to, students_info, messages.append, stamgroep_display)

    assert "Klas A: 2" in messages
    assert "Anne's groep: 1" in messages
    # The internal matching keys must not leak into the user-facing messages.
    assert not any("klasa" in m for m in messages)
