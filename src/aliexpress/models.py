"""Database models for the application.

Solve tracking
--------------
A process has at most one current run, keyed by ``process_id``; re-running a process resets
that row and its log lines. Log lines are kept in insertion order via their autoincrement
primary key. That id *is* the sequence number, and it is assigned atomically by the
database, so it stays race-free under the two background threads (solve and sociogram) that
append concurrently — SQLite serialises the writes. A manual counter would need a
read-then-write that those two threads could interleave.

Authentication
--------------
``School`` is both the SQLAlchemy model and the Flask-Login user object (it inherits
``UserMixin`` directly). There is no separate ``User`` wrapper: the school *is* the
authenticated entity, adding an indirection class would only obscure that.

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

    schoolcode = db.Column(db.String(64), primary_key=True)
    naam = db.Column(db.String(256), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def get_id(self):
        return self.schoolcode


class Run(db.Model):
    """The current solve run for one process."""

    process_id = db.Column(db.String, primary_key=True)
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


class LogLine(db.Model):
    """One user-facing progress line for a run, ordered by insertion (the primary key)."""

    id = db.Column(db.Integer, primary_key=True)
    process_id = db.Column(
        db.String, db.ForeignKey("run.process_id"), nullable=False, index=True
    )
    text = db.Column(db.Text, nullable=False)
