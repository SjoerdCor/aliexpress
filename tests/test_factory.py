"""Tests for the create_app() factory and the storage path helpers."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import pytest
from flask import Flask

import aliexpress as package
from aliexpress import create_app, create_reset_application
from aliexpress.web.extensions import db as _db
from aliexpress.web.models import School
from aliexpress.web.storage import get_file_path, get_process_path


@pytest.fixture()
def test_app(tmp_path):
    """Minimal app with an in-memory database and tmp storage dir."""
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "SQLALCHEMY_DATABASE_URI": "sqlite://",
            "STORAGE_DIR": str(tmp_path),
        }
    )


class TestCreateApp:
    """Tests for the create_app() application factory."""

    def test_returns_flask_app(self, test_app):
        """create_app() returns a configured Flask application object."""
        assert isinstance(test_app, Flask)
        assert test_app.config["TESTING"] is True

    def test_processing_poll_interval_defaults_to_one_second(self, test_app):
        """The test-only browser override must not change the production default."""
        assert test_app.config["PROCESSING_POLL_INTERVAL_MS"] == 1000

    def test_db_tables_are_queryable(self, test_app):
        """The database is initialised so that models can be queried."""
        with test_app.app_context():
            _db.create_all()
            assert School.query.count() == 0

    def test_reset_configuration_has_no_runtime_side_effects(
        self, monkeypatch, tmp_path
    ):
        """Inspecting reset targets does not create directories or initialize a DB."""
        monkeypatch.setattr(package, "_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("ALIEXPRESS_ENV", "local")
        monkeypatch.setenv("DATABASE_URL", "sqlite:///custom.db")

        application = create_reset_application()

        assert application.instance_path == str(tmp_path / "instance")
        assert application.config == {
            "ALIEXPRESS_ENV": "local",
            "SQLALCHEMY_DATABASE_URI": "sqlite:///custom.db",
            "STORAGE_DIR": str(tmp_path / "instance" / "storage"),
        }
        assert not (tmp_path / "instance").exists()


class TestStoragePaths:
    """Tests for the storage path helpers that read STORAGE_DIR from config."""

    def test_get_process_path_is_under_school_dir(self, test_app, tmp_path):
        """get_process_path returns <STORAGE_DIR>/<school>/<process>."""
        with test_app.app_context():
            result = get_process_path("school-1", "mijn-proces")

        assert result == str(tmp_path / "school-1" / "mijn-proces")

    def test_get_process_path_rejects_path_traversal(self, test_app):
        """get_process_path raises PermissionError when path escapes the school dir."""
        with test_app.app_context():
            try:
                get_process_path("school-1", "../andere-school/slecht")
                assert False, "Expected PermissionError for path traversal"
            except PermissionError:
                pass

    def test_get_file_path_appends_filename(self, test_app, tmp_path):
        """get_file_path returns <process_path>/<filename>."""
        with test_app.app_context():
            result = get_file_path("school-1", "mijn-proces", "results.xlsx")

        assert result == str(tmp_path / "school-1" / "mijn-proces" / "results.xlsx")
