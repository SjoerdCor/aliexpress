"""Tests for the separation between local and production configuration."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

from pathlib import Path

import pytest
from werkzeug.security import generate_password_hash

import aliexpress
from aliexpress import create_app
from aliexpress.web.extensions import db
from aliexpress.web.models import School

_TEST_SECRET_KEY = "test-secret-key"
_TEST_ADMIN_PASSWORD = "An-unpredictable-admin-test-password-42!"
_TEST_SCHOOLCODE = "test-school"
_TEST_SCHOOL_NAME = "Testschool"
_TEST_SCHOOL_PASSWORD = "test-school-password"


@pytest.fixture()
def app_settings(tmp_path):
    """Return safe settings for an isolated application instance."""
    return {
        "SECRET_KEY": _TEST_SECRET_KEY,
        "ADMIN_PASSWORD": _TEST_ADMIN_PASSWORD,
        "SQLALCHEMY_DATABASE_URI": "sqlite://",
        "STORAGE_DIR": str(tmp_path / "storage"),
    }


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ("local", {"debug": False, "testing": False, "secure_cookie": False}),
        ("production", {"debug": False, "testing": False, "secure_cookie": True}),
    ],
)
def test_environment_selects_explicit_application_config(
    monkeypatch, app_settings, environment, expected
):
    """Each supported environment has deliberate debug/testing/cookie settings."""
    monkeypatch.setenv("ALIEXPRESS_ENV", environment)
    monkeypatch.setenv("FLASK_ENV", "production")

    application = create_app(app_settings)

    assert application.config["DEBUG"] is expected["debug"]
    assert application.config["TESTING"] is expected["testing"]
    assert application.config["SESSION_COOKIE_SECURE"] is expected["secure_cookie"]
    assert application.config["USE_RELOADER"] is False


def test_testing_is_an_explicit_factory_override(monkeypatch, app_settings):
    """Tests opt into Flask's testing mode through ``create_app`` settings."""
    monkeypatch.setenv("ALIEXPRESS_ENV", "local")

    application = create_app({**app_settings, "TESTING": True})

    assert application.config["TESTING"] is True
    assert application.config["DEBUG"] is False
    assert application.config["SESSION_COOKIE_SECURE"] is False


@pytest.mark.parametrize("removed_environment", ["development", "testing"])
def test_removed_environment_names_are_rejected(
    monkeypatch, app_settings, removed_environment
):
    """Removed environment names must not silently select another configuration."""
    monkeypatch.setenv("ALIEXPRESS_ENV", removed_environment)

    with pytest.raises(RuntimeError, match="onbekende waarde"):
        create_app(app_settings)


def test_local_environment_is_not_overridden_by_flask_environment(
    monkeypatch, app_settings
):
    """The project environment variable, not Flask's legacy variable, is authoritative."""
    monkeypatch.setenv("ALIEXPRESS_ENV", "local")
    monkeypatch.setenv("FLASK_ENV", "production")

    application = create_app(app_settings)

    assert application.config["SESSION_COOKIE_SECURE"] is False


def test_local_environment_requires_an_explicit_secret_key(monkeypatch, app_settings):
    """Local sessions may never use a public, predictable fallback signing key."""
    monkeypatch.setenv("ALIEXPRESS_ENV", "local")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setattr(aliexpress, "load_dotenv", lambda **_kwargs: False)
    settings_without_secret = {
        key: value for key, value in app_settings.items() if key != "SECRET_KEY"
    }

    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app(settings_without_secret)


def test_unspecified_wsgi_environment_uses_secure_production_defaults(
    monkeypatch, app_settings
):
    """A WSGI factory call must not silently downgrade to the local HTTP config."""
    monkeypatch.delenv("ALIEXPRESS_ENV", raising=False)
    monkeypatch.setattr(aliexpress, "load_dotenv", lambda **_kwargs: False)

    application = create_app(app_settings)

    assert application.config["ALIEXPRESS_ENV"] == "production"
    assert application.config["SESSION_COOKIE_SECURE"] is True


def test_dotenv_is_resolved_from_project_root_when_cwd_changes(
    monkeypatch, app_settings, tmp_path
):
    """Changing directory must not make the factory look for a different .env file."""
    loaded_paths = []

    def record_dotenv_path(*, dotenv_path):
        loaded_paths.append(Path(dotenv_path))

    monkeypatch.setattr(aliexpress, "load_dotenv", record_dotenv_path)
    monkeypatch.chdir(tmp_path)

    create_app(app_settings)

    project_root = Path(aliexpress.__file__).resolve().parents[2]
    assert loaded_paths == [project_root / ".env"]


def test_local_login_session_survives_two_http_requests(monkeypatch, app_settings):
    """A local HTTP login cookie must be sent on the next request."""
    monkeypatch.setenv("ALIEXPRESS_ENV", "local")
    application = create_app(app_settings)

    with application.app_context():
        db.session.add(
            School(
                schoolcode=_TEST_SCHOOLCODE,
                naam=_TEST_SCHOOL_NAME,
                password_hash=generate_password_hash(_TEST_SCHOOL_PASSWORD),
            )
        )
        db.session.commit()

    with application.test_client() as client:
        login_response = client.post(
            "/login",
            data={
                "schoolcode": _TEST_SCHOOLCODE,
                "wachtwoord": _TEST_SCHOOL_PASSWORD,
            },
        )
        assert login_response.status_code == 302
        assert "Secure" not in login_response.headers["Set-Cookie"]

        protected_response = client.get("/processes")

    assert protected_response.status_code == 200
