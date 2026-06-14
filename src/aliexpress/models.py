"""Database models tracking the background solve for each process.

A process has at most one current run, keyed by ``process_id``; re-running a process resets
that row and its log lines. Log lines are kept in insertion order via their autoincrement
primary key. That id *is* the sequence number, and it is assigned atomically by the
database, so it stays race-free under the two background threads (solve and sociogram) that
append concurrently — SQLite serialises the writes. A manual counter would need a
read-then-write that those two threads could interleave.
"""

from datetime import datetime, timezone

from .extensions import db

# SQLAlchemy declarative models are attribute-only data classes; they legitimately have no
# public methods (same justification as the Flask config classes in appconfig.py).
# pylint: disable=too-few-public-methods


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
