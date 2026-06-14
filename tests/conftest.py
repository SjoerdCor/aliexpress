"""Shared pytest configuration.

Point the database at a throwaway file *before* the app (and its SQLAlchemy engine) are
imported, so the test suite never touches the real ``instance/app.db``. Individual tests
reset the tables for isolation; this only fixes where that database lives.
"""

import os
import tempfile

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
