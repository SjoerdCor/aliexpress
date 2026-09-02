"""Database models for the application.

Ownership chain
---------------
``School`` owns one or more ``Process`` instances (cascade delete). Each ``Process``
has at most one ``Run`` (1-to-1 via ``uselist=False``). Access control flows through
the chain: routes always look up ``Process`` by ``(school_id, name)`` — never by the
integer primary key — so a logged-in school cannot reach another school's data.

Solve tracking
--------------
A process has at most one current run. A new solve atomically claims that process and
reuses a completed or failed row. Progress during a run is reported via
``progress.json`` in the process directory (written by the solve thread), not through
the database.

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
from sqlalchemy import update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

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

    @classmethod
    def start_if_inactive(cls, process_id) -> bool:
        """Atomically claim a process for a fresh pending run.

        A missing row is claimed with SQLite's conflict-safe insert. Existing
        completed or failed rows are conditionally updated; active rows do not
        match that update and therefore remain unchanged.
        """
        insert_statement = sqlite_insert(cls).values(
            process_id=process_id, status="pending"
        )
        insert_statement = insert_statement.on_conflict_do_nothing(
            index_elements=[cls.process_id]
        )
        inserted = db.session.execute(insert_statement).rowcount == 1
        if inserted:
            db.session.commit()
            return True

        update_statement = (
            update(cls)
            .where(
                cls.process_id == process_id,
                cls.status.in_(("done", "error")),
            )
            .values(
                status="pending",
                message=None,
                created_at=datetime.now(timezone.utc),
            )
        )
        updated = db.session.execute(update_statement).rowcount == 1
        db.session.commit()
        return updated

    def set_status(self, status, message=None):
        """Persist a new status (and optional message) for this run."""
        self.status = status
        self.message = message
        db.session.commit()
