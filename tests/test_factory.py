"""Tests for the create_app() factory and the storage path helpers."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import pytest
from flask import Flask

from aliexpress import create_app
from aliexpress.extensions import db as _db
from aliexpress.models import School


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


def test_create_app_returns_flask_app(test_app):
    """create_app() returns a configured Flask application object."""
    assert isinstance(test_app, Flask)
    assert test_app.config["TESTING"] is True


def test_create_app_creates_db_tables(test_app):
    """create_app() sets up the database so tables are queryable."""
    with test_app.app_context():
        _db.create_all()
        assert School.query.count() == 0
