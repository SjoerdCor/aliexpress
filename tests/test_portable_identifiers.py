"""Acceptance tests for platform-independent identifiers and storage paths."""

# pylint: disable=redefined-outer-name,import-outside-toplevel

import unicodedata

import pytest

from aliexpress import create_app
from aliexpress.web.extensions import db, limiter
from aliexpress.web.identifiers import identifier_key
from aliexpress.web.models import Process, School
from aliexpress.web.storage import get_process_path, get_school_path
from app import app as flask_app
from tests.helpers import SCHOOL_ID, flashes


@pytest.fixture()
def portable_app(tmp_path):
    """Small isolated Flask app for pure storage-path tests."""
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "SQLALCHEMY_DATABASE_URI": "sqlite://",
            "STORAGE_DIR": str(tmp_path),
        }
    )
    with application.app_context():
        db.create_all()
    return application


@pytest.fixture()
def runner(tmp_path):
    """Isolated Flask CLI runner for school-code tests."""
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-key"
    flask_app.config["STORAGE_DIR"] = str(tmp_path)
    limiter.enabled = False
    with flask_app.app_context():
        db.drop_all()
        db.create_all()
    return flask_app.test_cli_runner()


@pytest.mark.parametrize(
    "value",
    [
        "naam/met/slash",
        r"naam\met\backslash",
        "../buiten",
        "..",
        "/absolute/pad",
        r"C:\absolute\pad",
    ],
)
def test_storage_rejects_separators_traversal_and_absolute_names(portable_app, value):
    """A name must remain one safe path segment on POSIX and Windows."""
    with portable_app.app_context():
        with pytest.raises(PermissionError):
            get_process_path("school-1", value)


@pytest.mark.parametrize("value", ["CON", "NUL", "COM1", "LPT1"])
def test_identifier_validation_rejects_windows_reserved_names(portable_app, value):
    """Windows device names must not become school or process directories."""
    with portable_app.app_context():
        with pytest.raises(PermissionError):
            get_school_path(value)


@pytest.mark.parametrize("value", ["CON", "NUL", "COM1", "LPT1"])
def test_process_storage_rejects_windows_reserved_names(portable_app, value):
    """The same reserved-name rule applies to process directories."""
    with portable_app.app_context():
        with pytest.raises(PermissionError):
            get_process_path("school-1", value)


def test_identifier_comparison_key_is_case_and_unicode_normalized():
    """Equivalent names have one comparison key across case and Unicode forms."""
    decomposed = "e\u0301cole"
    composed = unicodedata.normalize("NFC", decomposed)

    assert identifier_key("Klas") == identifier_key("klas")
    assert identifier_key(decomposed) == identifier_key(composed)


def test_identifier_maximum_length_is_enforced(portable_app):
    """Identifiers longer than the portable database/path limit are rejected."""
    with portable_app.app_context():
        with pytest.raises(PermissionError):
            get_school_path("a" * 65)


def test_symlinked_school_directory_cannot_escape_storage(portable_app, tmp_path):
    """An existing school directory that resolves outside storage is rejected."""
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    school_link = tmp_path / "school-1"
    school_link.symlink_to(outside, target_is_directory=True)

    with portable_app.app_context():
        with pytest.raises(PermissionError):
            get_school_path("school-1")


@pytest.mark.parametrize("schoolcode", ["CON", "NUL", "COM1", "LPT1", "../school"])
def test_admin_cli_rejects_nonportable_schoolcodes_before_database_mutation(
    runner, schoolcode
):
    """Invalid schoolcodes are rejected before a School row is added."""
    result = runner.invoke(
        args=["schools", "add", schoolcode, "--naam", "Ongeldige School"]
    )

    assert result.exit_code != 0
    assert "schoolcode" in result.output.lower()
    with flask_app.app_context():
        assert School.query.count() == 0


def test_admin_cli_treats_case_variants_as_the_same_schoolcode(runner):
    """The management CLI must prevent names colliding on case-insensitive filesystems."""
    first = runner.invoke(args=["schools", "add", "Klas", "--naam", "Klas School"])
    second = runner.invoke(args=["schools", "add", "klas", "--naam", "Andere School"])

    assert first.exit_code == 0, first.output
    assert second.exit_code != 0
    assert "bestaat al" in second.output


def test_admin_cli_treats_unicode_equivalents_as_the_same_schoolcode(runner):
    """NFC-equivalent schoolcodes must not create two storage identities."""
    decomposed = "e\u0301cole"
    composed = unicodedata.normalize("NFC", decomposed)
    first = runner.invoke(args=["schools", "add", decomposed, "--naam", "École"])
    second = runner.invoke(args=["schools", "add", composed, "--naam", "Andere"])

    assert first.exit_code == 0, first.output
    assert second.exit_code != 0
    assert "bestaat al" in second.output


def test_process_route_rejects_reserved_name_before_database_or_filesystem_mutation(
    client, tmp_path
):
    """A reserved process name must not leave a DB row or directory behind."""
    response = client.post("/processes/create", data={"process_name": "CON"})

    assert response.status_code == 302
    assert any("gereserveerde" in message for _, message in flashes(client))
    with flask_app.app_context():
        assert Process.query.filter_by(school_id=SCHOOL_ID).count() == 0
    assert not (tmp_path / SCHOOL_ID / "CON").exists()


def test_process_route_treats_case_variants_as_the_same_name(client):
    """A process cannot be duplicated using only case differences."""
    first = client.post("/processes/create", data={"process_name": "Klas"})
    second = client.post("/processes/create", data={"process_name": "klas"})

    assert first.status_code == 302
    assert second.status_code == 302
    assert flashes(client) == [("error", "Proces bestaat al")]
    with flask_app.app_context():
        process = Process.by_name(SCHOOL_ID, "klas")
        assert process.name == "Klas"


def test_process_route_treats_unicode_equivalents_as_the_same_name(client):
    """NFC-equivalent process names must map to one process directory."""
    decomposed = "e\u0301cole"
    composed = unicodedata.normalize("NFC", decomposed)
    first = client.post("/processes/create", data={"process_name": decomposed})
    second = client.post("/processes/create", data={"process_name": composed})

    assert first.status_code == 302
    assert second.status_code == 302
    assert flashes(client) == [("error", "Proces bestaat al")]


def test_process_route_rejects_overlong_name_without_mutating_storage(client, tmp_path):
    """A process name beyond the portable limit is rejected before creation."""
    response = client.post("/processes/create", data={"process_name": "a" * 65})

    assert response.status_code == 302
    assert any("maximaal" in message for _, message in flashes(client))
    with flask_app.app_context():
        assert Process.query.filter_by(school_id=SCHOOL_ID).count() == 0
    assert not (tmp_path / SCHOOL_ID).exists()
