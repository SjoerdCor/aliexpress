"""Tests for the guarded local-data reset and its server marker."""

# pylint: disable=redefined-outer-name

from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

import aliexpress.main as main_module
from aliexpress.main import main
from aliexpress.server_lock import lock_instance


@pytest.fixture()
def reset_application(tmp_path):
    """Return a local app-shaped config rooted entirely in a temporary directory."""
    instance_path = tmp_path / "instance"
    storage_dir = instance_path / "storage"
    storage_dir.mkdir(parents=True)
    return SimpleNamespace(
        instance_path=str(instance_path),
        config={
            "ALIEXPRESS_ENV": "local",
            "STORAGE_DIR": str(storage_dir),
            "SQLALCHEMY_DATABASE_URI": "sqlite:///app.db",
        },
    )


def _patch_reset_application(monkeypatch, application):
    """Make the console command use the isolated app without creating a real DB."""
    monkeypatch.setattr(
        main_module,
        "create_reset_application",
        lambda: application,
    )


def test_serve_locks_instance_before_application_creation(monkeypatch, tmp_path):
    """Server startup owns the instance before opening or creating its database."""
    instance_path = tmp_path / "instance"
    application = object()
    events = []

    def assert_instance_is_locked(event):
        with pytest.raises(click.ClickException, match="actief"):
            with lock_instance(instance_path):
                pass
        events.append(event)

    def fake_create_app(**_kwargs):
        assert_instance_is_locked("create")
        return application

    def fake_serve_foreground(received, host, port, open_browser):
        assert received is application
        assert (host, port, open_browser) == ("127.0.0.2", 43213, False)
        assert_instance_is_locked("serve")

    monkeypatch.setattr(main_module, "get_instance_path", lambda: instance_path)
    monkeypatch.setattr(main_module, "create_app", fake_create_app)
    monkeypatch.setattr(main_module, "serve_foreground", fake_serve_foreground)

    main_module.serve_configured("127.0.0.2", 43213, open_browser=False)

    assert events == ["create", "serve"]
    with lock_instance(instance_path):
        pass


def test_reset_local_data_confirms_resolved_targets_and_preserves_unrelated_files(
    monkeypatch, reset_application
):
    """A confirmed reset removes only the database and storage contents."""
    _patch_reset_application(monkeypatch, reset_application)
    instance_path = Path(reset_application.instance_path)
    storage_dir = Path(reset_application.config["STORAGE_DIR"])
    database = instance_path / "app.db"
    database.write_text("database")
    (storage_dir / "school-1").mkdir()
    (storage_dir / "school-1" / "result.xlsx").write_text("result")
    (instance_path / "logs").mkdir()
    (instance_path / "logs" / "aliexpress.log").write_text("log")
    (instance_path / "config.toml").write_text("config")
    (instance_path / "keep.txt").write_text("keep")

    result = CliRunner().invoke(main, ["reset-local-data"], input="y\n")

    assert result.exit_code == 0, result.output
    assert str(database.resolve()) in result.output
    assert str(storage_dir.resolve()) in result.output
    assert not database.exists()
    assert not list(storage_dir.iterdir())
    assert (instance_path / "logs" / "aliexpress.log").read_text() == "log"
    assert (instance_path / "config.toml").read_text() == "config"
    assert (instance_path / "keep.txt").read_text() == "keep"


def test_reset_local_data_cancel_keeps_all_targets(monkeypatch, reset_application):
    """Declining the confirmation must leave both the database and storage intact."""
    _patch_reset_application(monkeypatch, reset_application)
    instance_path = Path(reset_application.instance_path)
    storage_dir = Path(reset_application.config["STORAGE_DIR"])
    database = instance_path / "app.db"
    database.write_text("database")
    (storage_dir / "school-1").mkdir()
    (storage_dir / "school-1" / "result.xlsx").write_text("result")

    result = CliRunner().invoke(main, ["reset-local-data"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "Geannuleerd" in result.output
    assert database.read_text() == "database"
    assert (storage_dir / "school-1" / "result.xlsx").read_text() == "result"


def test_reset_local_data_yes_skips_confirmation(monkeypatch, reset_application):
    """The non-interactive flag permits scripted local cleanup."""
    _patch_reset_application(monkeypatch, reset_application)
    database = Path(reset_application.instance_path) / "app.db"
    database.write_text("database")

    result = CliRunner().invoke(main, ["reset-local-data", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Doorgaan" not in result.output
    assert not database.exists()


def test_reset_local_data_reports_an_empty_environment(monkeypatch, reset_application):
    """An empty local environment succeeds without asking for confirmation."""
    _patch_reset_application(monkeypatch, reset_application)

    result = CliRunner().invoke(main, ["reset-local-data"])

    assert result.exit_code == 0, result.output
    assert "leeg" in result.output.lower()
    assert "Doorgaan" not in result.output


def test_reset_local_data_refuses_an_active_instance(monkeypatch, reset_application):
    """A held instance lock blocks deletion before confirmation or file changes."""
    _patch_reset_application(monkeypatch, reset_application)
    instance_path = Path(reset_application.instance_path)
    database = instance_path / "app.db"
    database.write_text("database")
    with lock_instance(instance_path):
        result = CliRunner().invoke(main, ["reset-local-data", "--yes"])

    assert result.exit_code != 0
    assert "actief" in result.output.lower()
    assert database.exists()


def test_reset_local_data_rejects_database_symlink(monkeypatch, reset_application):
    """A configured database symlink may never redirect deletion to another file."""
    _patch_reset_application(monkeypatch, reset_application)
    instance_path = Path(reset_application.instance_path)
    log_dir = instance_path / "logs"
    log_dir.mkdir()
    log = log_dir / "aliexpress.log"
    log.write_text("must remain")
    database = instance_path / "app.db"
    try:
        database.symlink_to(log)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    result = CliRunner().invoke(main, ["reset-local-data", "--yes"])

    assert result.exit_code != 0
    assert "symlink" in result.output.lower()
    assert log.read_text() == "must remain"
    assert database.is_symlink()


def test_reset_local_data_validates_all_sidecars_before_deleting(
    monkeypatch, reset_application
):
    """An unexpected sidecar type must not leave a partially deleted database."""
    _patch_reset_application(monkeypatch, reset_application)
    instance_path = Path(reset_application.instance_path)
    database = instance_path / "app.db"
    database.write_text("database")
    (instance_path / "app.db-wal").mkdir()

    result = CliRunner().invoke(main, ["reset-local-data", "--yes"])

    assert result.exit_code != 0
    assert "niets verwijderd" in result.output.lower()
    assert database.read_text() == "database"


@pytest.mark.parametrize("database_uri", [None, ""])
def test_reset_local_data_reports_malformed_database_urls(
    monkeypatch, reset_application, database_uri
):
    """Malformed database configuration becomes a controlled Click error."""
    _patch_reset_application(monkeypatch, reset_application)
    reset_application.config["SQLALCHEMY_DATABASE_URI"] = database_uri

    result = CliRunner().invoke(main, ["reset-local-data", "--yes"])

    assert result.exit_code != 0
    assert "database_url" in result.output.lower()
    assert isinstance(result.exception, SystemExit)


def test_reset_local_data_refuses_non_local_database_without_touching_it(
    monkeypatch, reset_application, tmp_path
):
    """An external database URL is rejected and is never interpreted as a file path."""
    _patch_reset_application(monkeypatch, reset_application)
    external_database = tmp_path / "external.db"
    external_database.write_text("must remain")
    reset_application.config["SQLALCHEMY_DATABASE_URI"] = (
        "postgresql://user:password@example.test/aliexpress"
    )
    reset_application.config["EXTERNAL_DATABASE_FILE"] = str(external_database)

    result = CliRunner().invoke(main, ["reset-local-data", "--yes"])

    assert result.exit_code != 0
    assert "sqlite" in result.output.lower()
    assert external_database.read_text() == "must remain"
