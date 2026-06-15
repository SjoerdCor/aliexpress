"""Root conftest: set env vars before the app is imported anywhere in the suite.

pytest loads root conftest.py before subdirectory conftest files, so this runs
before tests/conftest.py imports the app. DATABASE_URL keeps the suite off the real
database; SECRET_KEY satisfies the startup guard that refuses to run without one.
"""

import os
import tempfile

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
