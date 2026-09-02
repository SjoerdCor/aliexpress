"""Tests for the new convenience methods added to Process and Run models."""

# pylint: disable=redefined-outer-name,unused-argument  # standard pytest fixture patterns

from datetime import datetime, timezone

import pytest
from werkzeug.security import generate_password_hash

from aliexpress import create_app
from aliexpress.web.extensions import db as _db
from aliexpress.web.models import Process, Run, School


@pytest.fixture()
def app_ctx(tmp_path):
    """App with fresh in-memory DB; yields inside an app context."""
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "SQLALCHEMY_DATABASE_URI": "sqlite://",
            "STORAGE_DIR": str(tmp_path),
        }
    )
    with app.app_context():
        _db.create_all()
        yield app


@pytest.fixture()
def school(app_ctx):
    """Persisted School row for use in process tests."""
    s = School(
        schoolcode="school-a",
        naam="School A",
        password_hash=generate_password_hash("x"),
    )
    _db.session.add(s)
    _db.session.commit()
    return s


@pytest.fixture()
def process(school):
    """Persisted Process row owned by the test school."""
    p = Process(school_id=school.schoolcode, name="test-proces")
    _db.session.add(p)
    _db.session.commit()
    return p


class TestProcessByName:
    """Tests for Process.by_name() classmethod."""

    def test_returns_process_when_found(self, process):
        """by_name returns the matching Process for known school + name."""
        result = Process.by_name("school-a", "test-proces")
        assert result is not None
        assert result.id == process.id

    def test_returns_none_for_unknown_name(self, school):
        """by_name returns None when the process name does not exist."""
        result = Process.by_name("school-a", "bestaat-niet")
        assert result is None

    def test_returns_none_for_wrong_school(self, process):
        """by_name returns None when school_id does not match."""
        result = Process.by_name("andere-school", "test-proces")
        assert result is None


class TestRunStartIfInactive:
    """Tests for the atomic Run.start_if_inactive() classmethod."""

    def test_claims_process_without_existing_run(self, process):
        """A process without a run can be claimed and receives pending status."""
        assert Run.start_if_inactive(process.id) is True

        _db.session.refresh(process)
        assert process.run is not None
        assert process.run.status == "pending"

    @pytest.mark.parametrize("status", ["done", "error"])
    def test_reclaims_inactive_run(self, process, status):
        """A completed or failed run is reset to a fresh pending claim."""
        old_run = Run(
            process_id=process.id,
            status=status,
            message="oude foutmelding",
            created_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        _db.session.add(old_run)
        _db.session.commit()
        old_created_at = old_run.created_at

        assert Run.start_if_inactive(process.id) is True

        _db.session.refresh(old_run)
        assert old_run.status == "pending"
        assert old_run.message is None
        assert old_run.created_at > old_created_at

    @pytest.mark.parametrize("status", ["pending", "running"])
    def test_rejects_active_run_without_changing_it(self, process, status):
        """An active run cannot be claimed and remains fully intact."""
        active_run = Run(
            process_id=process.id,
            status=status,
            message="actieve melding",
            created_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        _db.session.add(active_run)
        _db.session.commit()
        old_created_at = active_run.created_at

        assert Run.start_if_inactive(process.id) is False

        _db.session.refresh(active_run)
        assert active_run.status == status
        assert active_run.message == "actieve melding"
        assert active_run.created_at == old_created_at


class TestRunSetStatus:
    """Tests for Run.set_status() instance method."""

    def test_updates_status(self, process):
        """set_status() persists the new status to the database."""
        run = Run(process_id=process.id)
        _db.session.add(run)
        _db.session.commit()

        run.set_status("running")

        _db.session.expire(run)
        assert run.status == "running"
        assert run.message is None

    def test_updates_status_and_message(self, process):
        """set_status() with a message persists both fields."""
        run = Run(process_id=process.id)
        _db.session.add(run)
        _db.session.commit()

        run.set_status("error", "Iets ging mis")

        _db.session.expire(run)
        assert run.status == "error"
        assert run.message == "Iets ging mis"
