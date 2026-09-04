"""Tests for the platform-neutral ``ali-express`` console command."""

import importlib.metadata

from click.testing import CliRunner

import aliexpress.main as main_module
from aliexpress.main import main


def _registered_entry_point(name):
    return next(
        entry_point
        for entry_point in importlib.metadata.entry_points(group="console_scripts")
        if entry_point.name == name
    )


def test_registered_console_entrypoints_load_the_same_command():
    """Both published spellings must resolve to the canonical Click command."""
    assert _registered_entry_point("ali-express").load() is main
    assert _registered_entry_point("aliexpress").load() is main


def test_help_lists_the_solve_command():
    """The canonical command must be discoverable without touching application data."""
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "solve" in result.output


def test_solve_delegates_to_existing_solver(monkeypatch):
    """``solve`` must reuse the established file-reading/distribution function."""
    called = []

    def fake_distribute_students_once():
        called.append(True)

    monkeypatch.setattr(
        main_module, "distribute_students_once", fake_distribute_students_once
    )

    result = CliRunner().invoke(main, ["solve"])

    assert result.exit_code == 0, result.output
    assert called == [True]
