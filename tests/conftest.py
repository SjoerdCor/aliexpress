"""Shared pytest fixtures.

DATABASE_URL is set in the root conftest.py before this file is loaded, so all
imports here can be at the top without ordering constraints.
"""

import pytest
from werkzeug.security import generate_password_hash

from aliexpress.web.extensions import db, limiter
from aliexpress.web.models import School
from app import app as flask_app


@pytest.fixture()
def client(tmp_path):
    """Flask test client logged in as a test school; database tables reset per test.

    Almost all routes are protected, so logging in by default avoids duplicating
    auth setup in every test. Use ``unauthed_client`` to test unauthenticated behaviour.
    """
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-key"
    flask_app.config["STORAGE_DIR"] = str(tmp_path)
    # The shared client logs in for every test; without this the suite would trip the
    # login rate limit. flask-limiter reads RATELIMIT_ENABLED only at init, so toggle the
    # live attribute. One dedicated test re-enables it to cover the limiter itself.
    limiter.enabled = False
    with flask_app.app_context():
        db.drop_all()
        db.create_all()
        school = School(
            schoolcode="test-school",
            naam="Testschool",
            password_hash=generate_password_hash("testpass"),
        )
        db.session.add(school)
        db.session.commit()
    with flask_app.test_client() as c:
        c.post("/login", data={"schoolcode": "test-school", "wachtwoord": "testpass"})
        yield c


@pytest.fixture()
def unauthed_client(tmp_path):
    """Flask test client without a logged-in session; database tables reset per test."""
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-key"
    flask_app.config["STORAGE_DIR"] = str(tmp_path)
    limiter.enabled = False
    with flask_app.app_context():
        db.drop_all()
        db.create_all()
    with flask_app.test_client() as c:
        yield c
