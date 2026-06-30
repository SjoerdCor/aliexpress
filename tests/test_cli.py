"""Tests for Flask CLI commands in src/aliexpress/cli.py.

The ``schools`` and ``admins`` command groups manage school accounts and admin
accounts directly from the command line — operations that bypass the normal HTTP
layer and act on the database and filesystem without a confirmation from the web
UI.  Because the commands are destructive (deletes wipe both DB rows and stored
files) and security-sensitive (admin creation), wrong behaviour here can cause
permanent data loss or allow weak credentials.  These tests use Flask's
``test_cli_runner`` to exercise the commands against an isolated in-memory DB and
a temporary filesystem, so no real data is touched.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import pytest
from werkzeug.security import generate_password_hash

from aliexpress.web.extensions import db, limiter
from aliexpress.web.models import Admin, School
from app import app as flask_app


@pytest.fixture()
def runner(tmp_path):
    """CLI test runner with a fresh DB and a temporary STORAGE_DIR.

    Mirrors the setup used by the ``client`` and ``unauthed_client`` fixtures in
    conftest.py so the CLI commands see the same isolation guarantees as HTTP tests.
    """
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-key"
    flask_app.config["STORAGE_DIR"] = str(tmp_path)
    limiter.enabled = False
    with flask_app.app_context():
        db.drop_all()
        db.create_all()
    return flask_app.test_cli_runner()


def _create_school(schoolcode="obs-test", naam="Testschool", password="tijdelijk"):
    """Persist a School in the current app context and return it."""
    with flask_app.app_context():
        school = School(
            schoolcode=schoolcode,
            naam=naam,
            password_hash=generate_password_hash(password),
        )
        db.session.add(school)
        db.session.commit()
    return schoolcode


class TestSchoolsDeleteCommand:
    """Guards the ``schools delete`` command: DB row removal, storage cleanup, abort, unknown."""

    def test_happy_path_removes_db_row_and_storage_dir(self, runner, tmp_path):
        """Confirming deletion must remove the School row AND its directory under
        STORAGE_DIR.  Both must be gone: an orphaned directory wastes disk and may
        contain student data; an orphaned DB row blocks re-creating the school."""
        schoolcode = _create_school()
        school_dir = tmp_path / schoolcode
        school_dir.mkdir()
        (school_dir / "result.xlsx").write_text("dummy")

        result = runner.invoke(args=["schools", "delete", schoolcode], input="y\n")

        assert result.exit_code == 0, result.output
        with flask_app.app_context():
            assert db.session.get(School, schoolcode) is None
        assert not school_dir.exists()

    def test_abort_keeps_db_row_and_storage_dir(self, runner, tmp_path):
        """Answering 'n' to the confirmation prompt must abort the command without
        touching either the DB row or the storage directory.  This guards the
        ``abort=True`` safety valve: a fat-finger 'n' must leave everything intact."""
        schoolcode = _create_school()
        school_dir = tmp_path / schoolcode
        school_dir.mkdir()
        (school_dir / "result.xlsx").write_text("dummy")

        result = runner.invoke(args=["schools", "delete", schoolcode], input="n\n")

        assert result.exit_code != 0
        with flask_app.app_context():
            assert db.session.get(School, schoolcode) is not None
        assert school_dir.exists()

    def test_unknown_schoolcode_yields_error_message(self, runner):
        """Attempting to delete a non-existent school must fail with a user-facing Dutch
        error message containing 'bestaat niet'.  Nothing should be deleted because
        there is nothing to delete — but the exit code must be non-zero so scripts can
        detect the failure."""
        result = runner.invoke(args=["schools", "delete", "bestaat-niet-school"])

        assert result.exit_code != 0
        assert "bestaat niet" in result.output


class TestSchoolsAddCommand:
    """Guards the ``schools add`` command: DB row creation and temporary-password output."""

    def test_happy_path_creates_school_with_must_change_password(self, runner):
        """A new school created via the CLI must have ``must_change_password=True`` so
        the temporary password cannot be used indefinitely.  The command must also echo
        the temporary password so the admin can hand it to the school — if it is never
        shown, the school cannot log in at all."""
        result = runner.invoke(
            args=["schools", "add", "obs-nieuw", "--naam", "Nieuwe School"]
        )

        assert result.exit_code == 0, result.output
        assert "Tijdelijk wachtwoord" in result.output
        with flask_app.app_context():
            school = db.session.get(School, "obs-nieuw")
            assert school is not None
            assert school.must_change_password is True

    def test_duplicate_schoolcode_raises_error(self, runner):
        """Adding a school whose code already exists must fail with a non-zero exit
        code.  Without this guard a second ``schools add obs-nieuw`` would overwrite
        the existing row's hash, locking the original school out of its account."""
        _create_school(schoolcode="obs-dup")
        result = runner.invoke(
            args=["schools", "add", "obs-dup", "--naam", "Duplicaat School"]
        )

        assert result.exit_code != 0
        assert "bestaat al" in result.output


class TestAdminsAddCommand:
    """Guards the ``admins add`` command: weak-password rejection and DB row creation."""

    def test_weak_password_is_rejected_and_no_admin_row_created(self, runner):
        """A trivially weak password (zxcvbn score < 3) must be rejected with a
        non-zero exit and a Dutch error message.  No Admin row must be written to the
        DB: if the row were written before the score check, a buggy implementation
        could let a weak password slip through and leave a permanent low-security
        admin account behind."""
        result = runner.invoke(
            args=["admins", "add", "newadmin"], input="weakpw\nweakpw\n"
        )

        assert result.exit_code != 0
        assert "zwak" in result.output.lower() or "Wachtwoord" in result.output
        with flask_app.app_context():
            assert Admin.query.filter_by(username="newadmin").first() is None

    def test_strong_password_creates_admin_row(self, runner):
        """A strong passphrase (zxcvbn score >= 3) must create an Admin row and exit
        cleanly.  This is the happy path; without it the command is useless and the
        application can never have an admin account."""
        strong = "PaardenBloem!Fiets42Oost"
        result = runner.invoke(
            args=["admins", "add", "superadmin"], input=f"{strong}\n{strong}\n"
        )

        assert result.exit_code == 0, result.output
        with flask_app.app_context():
            admin = Admin.query.filter_by(username="superadmin").first()
            assert admin is not None

    def test_duplicate_username_raises_error(self, runner):
        """Adding an admin whose username already exists must fail with a non-zero exit
        and a Dutch 'bestaat al' message.  Without this check a second invocation would
        silently overwrite the hash of an existing admin account."""
        strong = "PaardenBloem!Fiets42Oost"
        runner.invoke(
            args=["admins", "add", "bestaande"], input=f"{strong}\n{strong}\n"
        )
        result = runner.invoke(
            args=["admins", "add", "bestaande"], input=f"{strong}\n{strong}\n"
        )

        assert result.exit_code != 0
        assert "bestaat al" in result.output
