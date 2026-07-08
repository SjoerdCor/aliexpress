"""Tests for seeding the sole admin account from ADMIN_PASSWORD.

``seed_admin_from_env`` creates the admin on first start and rotates its password
hash when ADMIN_PASSWORD changes, but must never rehash (and commit) on every
restart when the password is unchanged. ``ensure_admin_password`` is the fail-fast
guard that keeps the app from starting with a missing or weak password.
"""

import pytest
from werkzeug.security import check_password_hash

from aliexpress.web.admin_seed import ensure_admin_password, seed_admin_from_env
from aliexpress.web.models import Admin

ADMIN_PASSWORD = "AdminGeheim!42xyz"
OTHER_ADMIN_PASSWORD = "AndersGeheim!73abc"


class TestSeedAdminFromEnv:
    """seed_admin_from_env creates, rotates, or leaves alone the sole admin row."""

    def test_creates_admin_when_none_exists(self, unauthed_client):
        """With no Admin row yet, seeding creates one matching ADMIN_PASSWORD."""
        app = unauthed_client.application
        with app.app_context():
            app.config["ADMIN_PASSWORD"] = ADMIN_PASSWORD
            seed_admin_from_env(app)
            admin = Admin.query.first()
            assert admin is not None
            assert check_password_hash(admin.password_hash, ADMIN_PASSWORD)

    def test_rotates_hash_when_password_changed(self, unauthed_client):
        """A changed ADMIN_PASSWORD updates the existing admin's hash in place."""
        app = unauthed_client.application
        with app.app_context():
            app.config["ADMIN_PASSWORD"] = ADMIN_PASSWORD
            seed_admin_from_env(app)

            app.config["ADMIN_PASSWORD"] = OTHER_ADMIN_PASSWORD
            seed_admin_from_env(app)

            assert Admin.query.count() == 1
            admin = Admin.query.first()
            assert not check_password_hash(admin.password_hash, ADMIN_PASSWORD)
            assert check_password_hash(admin.password_hash, OTHER_ADMIN_PASSWORD)

    def test_no_op_when_password_unchanged(self, unauthed_client):
        """Re-seeding with the same password does not rehash (proven by identical hash)."""
        app = unauthed_client.application
        with app.app_context():
            app.config["ADMIN_PASSWORD"] = ADMIN_PASSWORD
            seed_admin_from_env(app)
            hash_before = Admin.query.first().password_hash

            seed_admin_from_env(app)
            hash_after = Admin.query.first().password_hash

            assert hash_before == hash_after


class TestEnsureAdminPassword:
    """ensure_admin_password is the fail-fast startup guard."""

    def test_raises_when_password_missing(self, unauthed_client):
        """A missing ADMIN_PASSWORD refuses to start."""
        app = unauthed_client.application
        app.config["ADMIN_PASSWORD"] = None
        with pytest.raises(RuntimeError):
            ensure_admin_password(app)

    def test_raises_when_password_empty(self, unauthed_client):
        """An empty-string ADMIN_PASSWORD refuses to start."""
        app = unauthed_client.application
        app.config["ADMIN_PASSWORD"] = ""
        with pytest.raises(RuntimeError):
            ensure_admin_password(app)

    def test_raises_when_password_weak(self, unauthed_client):
        """A weak (low zxcvbn score) ADMIN_PASSWORD refuses to start."""
        app = unauthed_client.application
        app.config["ADMIN_PASSWORD"] = "password"
        with pytest.raises(RuntimeError):
            ensure_admin_password(app)

    def test_accepts_strong_password(self, unauthed_client):
        """A strong ADMIN_PASSWORD passes without raising."""
        app = unauthed_client.application
        app.config["ADMIN_PASSWORD"] = ADMIN_PASSWORD
        ensure_admin_password(app)
