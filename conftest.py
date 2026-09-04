"""Root conftest: set env vars before the app is imported anywhere in the suite.

pytest loads root conftest.py before subdirectory conftest files, so this runs
before tests/conftest.py imports the app. DATABASE_URL keeps the suite off the real
database; SECRET_KEY and ADMIN_PASSWORD satisfy the startup guards that refuse to
run without them.
"""

import os
import tempfile
import warnings

# Imports stay local so this module can establish DATABASE_URL before the application
# is imported by the rest of the test suite.
# pylint: disable=import-outside-toplevel

_worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
_db_fd, _db_path = tempfile.mkstemp(
    prefix=f"aliexpress-pytest-{_worker}-",
    suffix=".db",
)
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("ADMIN_PASSWORD", "AdminGeheim!42xyz")


def _cleanup_test_database():
    """Close SQLAlchemy resources and remove this process's temporary database."""
    errors = []

    try:
        from aliexpress.web.extensions import db
        from app import app

        with app.app_context():
            db.session.remove()
            for engine in db.engines.values():
                engine.dispose()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        errors.append(f"closing SQLAlchemy resources: {exc}")

    try:
        os.remove(_db_path)
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - platform-dependent cleanup failure
        errors.append(f"removing {_db_path}: {exc}")

    return errors


def pytest_sessionfinish(session, exitstatus):
    """Best-effort cleanup without hiding an earlier test failure."""
    del session
    errors = _cleanup_test_database()
    if errors and exitstatus == 0:
        warnings.warn(
            f"Could not clean up pytest database {_db_path}: {'; '.join(errors)}",
            UserWarning,
            stacklevel=2,
        )
