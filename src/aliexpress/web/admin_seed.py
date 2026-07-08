"""Seed the sole admin account from an environment variable.

Admin credentials are managed as configuration (``ADMIN_PASSWORD`` in ``.env``), not
through a signup flow: there is exactly one admin account, and its password is
rotated by editing the environment and restarting the app. ``ensure_admin_password``
fails fast at startup if that variable is missing or too weak; ``seed_admin_from_env``
creates or updates the ``Admin`` row to match it.
"""

from werkzeug.security import check_password_hash, generate_password_hash
from zxcvbn import zxcvbn

from .extensions import db
from .models import Admin


def ensure_admin_password(flask_app):
    """Refuse to start without a strong ADMIN_PASSWORD.

    An empty ADMIN_PASSWORD would leave the admin account unset or guessable, so fail
    fast at startup rather than silently skipping the admin seed or accepting a weak
    password. The zxcvbn score-3 threshold mirrors the one enforced for admin accounts
    created via ``flask admins add`` (see ``web/cli.py``).
    """
    password = flask_app.config.get("ADMIN_PASSWORD")
    if not password:
        raise RuntimeError(
            "ADMIN_PASSWORD is not set; refusing to start. Set it in the environment (.env)."
        )
    if zxcvbn(password)["score"] < 3:
        raise RuntimeError(
            "ADMIN_PASSWORD is too weak; refusing to start. Choose a stronger password."
        )


def seed_admin_from_env(flask_app):
    """Create or update the sole admin account to match ADMIN_PASSWORD.

    Must be called inside an application context. Creates the admin if none exists;
    rotates the stored hash if ADMIN_PASSWORD changed since the last start; otherwise
    does nothing, so a restart with an unchanged password never rehashes or commits.
    """
    password = flask_app.config["ADMIN_PASSWORD"]
    admin = Admin.query.first()
    if admin is None:
        admin = Admin(password_hash=generate_password_hash(password))
        db.session.add(admin)
        db.session.commit()
    elif not check_password_hash(admin.password_hash, password):
        admin.password_hash = generate_password_hash(password)
        db.session.commit()
