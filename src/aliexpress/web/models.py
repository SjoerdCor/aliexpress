"""Database models for the application.

Ownership chain
---------------
``School`` owns one or more ``Process`` instances (cascade delete). Each ``Process``
has at most one ``Run`` (1-to-1 via ``uselist=False``). Access control flows through
the chain: routes always look up ``Process`` by ``(school_id, name)`` — never by the
integer primary key — so a logged-in school cannot reach another school's data.

Solve tracking
--------------
A process has at most one current run. Re-running a process resets that row and its
log lines. Log lines are kept in insertion order via their autoincrement primary key,
which is assigned atomically by the database and stays race-free under the two background
threads (solve and sociogram) that append concurrently.

Authentication
--------------
``School`` is both the SQLAlchemy model and the Flask-Login user object (it inherits
``UserMixin`` directly). There is no separate ``User`` wrapper: the school *is* the
authenticated entity.

``get_id()`` is overridden to return ``schoolcode`` (the primary key) because Flask-Login
calls ``get_id()`` to store the identity in the session and passes that string back to
``user_loader``.
"""

from datetime import datetime, timezone

from flask_login import UserMixin

from .extensions import db

# SQLAlchemy declarative models are attribute-only data classes; they legitimately have no
# public methods (same justification as the Flask config classes in appconfig.py).
# pylint: disable=too-few-public-methods


class School(UserMixin, db.Model):
    """A school that can log in to the application.

    ``schoolcode`` is both the primary key and the Flask-Login identity (returned by
    ``get_id()``). Passwords are stored as Werkzeug hashes — never plain text.
    """

    is_admin = False

    schoolcode = db.Column(db.String(64), primary_key=True)
    naam = db.Column(db.String(256), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    must_change_password = db.Column(
        db.Boolean, nullable=False, server_default="0", default=False
    )
    processes = db.relationship(
        "Process", backref="school", cascade="all, delete-orphan"
    )

    def get_id(self):
        return self.schoolcode


class Admin(UserMixin, db.Model):
    """An administrator account that can view and act on behalf of any school.

    Flask-Login identity is ``"admin:<id>"`` — the ``"admin:"`` prefix prevents collisions
    with school codes in the shared ``user_loader``.
    """

    is_admin = True

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    password_hash = db.Column(db.String(256), nullable=False)

    @property
    def naam(self):
        """Display name used in templates, mirrors the School.naam column."""
        return "Beheerder"

    def get_id(self):
        return f"admin:{self.id}"


class Process(db.Model):
    """A distribution process owned by one school.

    ``name`` is the user-visible process name (unique per school). The integer ``id``
    is an internal primary key used only by ``Run`` as a foreign key — no route exposes
    or accepts it as input.
    """

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    school_id = db.Column(
        db.String(64), db.ForeignKey("school.schoolcode"), nullable=False, index=True
    )
    name = db.Column(db.String, nullable=False)
    created_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    run = db.relationship(
        "Run", backref="process", uselist=False, cascade="all, delete-orphan"
    )
    __table_args__ = (
        db.UniqueConstraint("school_id", "name", name="uq_process_school_name"),
    )

    @classmethod
    def by_name(cls, school_id, name):
        """Return the Process for this school + name, or None when not found."""
        return cls.query.filter_by(school_id=school_id, name=name).first()


class Run(db.Model):
    """The current solve run for one process (1-to-1 with Process)."""

    process_id = db.Column(db.Integer, db.ForeignKey("process.id"), primary_key=True)
    status = db.Column(db.String, nullable=False, default="pending")
    message = db.Column(db.Text)
    created_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    log_lines = db.relationship(
        "LogLine",
        backref="run",
        cascade="all, delete-orphan",
        order_by="LogLine.id",
    )

    @classmethod
    def reset(cls, process_id):
        """Replace any existing run for this process with a fresh pending run.

        Deletes the old row first (cascading to log lines) so the new run starts
        clean. Commits when done; callers in background threads open their own
        session and will see the new row.
        """
        existing = db.session.get(cls, process_id)
        if existing is not None:
            db.session.delete(existing)
            db.session.flush()
        db.session.add(cls(process_id=process_id))
        db.session.commit()

    def set_status(self, status, message=None):
        """Persist a new status (and optional message) for this run."""
        self.status = status
        self.message = message
        db.session.commit()


class LogLine(db.Model):
    """One user-facing progress line for a run, ordered by insertion (the primary key)."""

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(
        db.Integer, db.ForeignKey("run.process_id"), nullable=False, index=True
    )
    text = db.Column(db.Text, nullable=False)
