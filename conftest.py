"""Root conftest: set DATABASE_URL before the app is imported anywhere in the suite.

pytest loads root conftest.py before subdirectory conftest files, so this runs
before tests/conftest.py imports the app (which would otherwise create the real DB).
"""

import os
import tempfile

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
