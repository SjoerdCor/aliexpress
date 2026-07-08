"""Tests for the admin blueprint and school impersonation.

These cover the access boundary that lets an admin act on behalf of any school while
keeping non-admins out: the admin-only guard, admin login, the dashboard, the
impersonate / stop-impersonating flow, and the auth helpers (``load_user`` and the
``effective_school_id``-driven routing) that send an admin's requests to the right place.
A regression here would either lock the admin out or, worse, let a non-admin reach
another school's data.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import pytest
from werkzeug.security import generate_password_hash

from aliexpress.web.admin_seed import seed_admin_from_env
from aliexpress.web.extensions import db
from aliexpress.web.models import Admin, Process, School
from aliexpress.web.routes.auth import load_user

ADMIN_PASSWORD = "AdminGeheim!42xyz"


def _create_admin(app, password=ADMIN_PASSWORD):
    """Seed an Admin via seed_admin_from_env and return its integer id."""
    with app.app_context():
        app.config["ADMIN_PASSWORD"] = password
        seed_admin_from_env(app)
        return Admin.query.first().id


def _create_school(app, schoolcode="obs-x", naam="School X", password="schoolpw"):
    """Persist a School and return its schoolcode."""
    with app.app_context():
        db.session.add(
            School(
                schoolcode=schoolcode,
                naam=naam,
                password_hash=generate_password_hash(password),
            )
        )
        db.session.commit()
    return schoolcode


@pytest.fixture()
def admin_client(unauthed_client):
    """An unauthenticated client with an admin account created and logged in."""
    _create_admin(unauthed_client.application)
    unauthed_client.post(
        "/admin/login",
        data={"wachtwoord": ADMIN_PASSWORD},
    )
    return unauthed_client


class TestAdminRequired:
    """The _admin_required guard keeps non-admins out of admin routes."""

    def test_anonymous_is_redirected_to_admin_login(self, unauthed_client):
        """An unauthenticated visitor to an admin route is sent to the admin login."""
        resp = unauthed_client.get("/admin/")
        assert resp.status_code == 302
        assert "/admin/login" in resp.headers["Location"]

    def test_school_user_is_redirected_to_admin_login(self, client):
        """A logged-in school (not an admin) cannot reach the admin dashboard."""
        resp = client.get("/admin/")
        assert resp.status_code == 302
        assert "/admin/login" in resp.headers["Location"]


class TestAdminLogin:
    """Admin login form: rendering, rejection, success, and the already-in redirect."""

    def test_get_renders_login(self, unauthed_client):
        """GET /admin/login is reachable without authentication."""
        assert unauthed_client.get("/admin/login").status_code == 200

    def test_wrong_credentials_re_render_with_error(self, unauthed_client):
        """Wrong admin credentials re-render the form with the Dutch error."""
        _create_admin(unauthed_client.application)
        resp = unauthed_client.post(
            "/admin/login",
            data={"wachtwoord": "fout"},
        )
        assert resp.status_code == 200
        assert "Ongeldig wachtwoord" in resp.data.decode("utf-8")

    def test_correct_credentials_redirect_to_dashboard(self, unauthed_client):
        """Correct admin credentials log in and redirect to the dashboard."""
        _create_admin(unauthed_client.application)
        resp = unauthed_client.post(
            "/admin/login",
            data={"wachtwoord": ADMIN_PASSWORD},
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/admin/")

    def test_already_logged_in_admin_skips_login(self, admin_client):
        """An already-authenticated admin hitting /admin/login goes straight to the dashboard."""
        resp = admin_client.get("/admin/login")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/admin/")


class TestAdminDashboard:
    """The dashboard lists every school and its processes."""

    def test_dashboard_lists_schools_and_processes(self, admin_client):
        """A school and its process appear on the dashboard (the school/process loop runs)."""
        app = admin_client.application
        _create_school(app, schoolcode="obs-x", naam="School X")
        with app.app_context():
            db.session.add(Process(school_id="obs-x", name="indeling-2026"))
            db.session.commit()
        resp = admin_client.get("/admin/")
        assert resp.status_code == 200
        body = resp.data.decode("utf-8")
        assert "School X" in body
        assert "indeling-2026" in body

    def test_dashboard_renders_with_no_schools(self, admin_client):
        """The dashboard renders (200) even when no schools exist yet (empty loop)."""
        assert admin_client.get("/admin/").status_code == 200


class TestImpersonation:
    """Impersonation lets an admin act as a school; the auth helpers route accordingly."""

    def test_impersonate_sets_session_and_redirects_to_processes(self, admin_client):
        """Impersonating a school stores it in the session and lands on its processes."""
        _create_school(admin_client.application, schoolcode="obs-x")
        resp = admin_client.get("/admin/impersonate/obs-x")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/processes")
        with admin_client.session_transaction() as sess:
            assert sess["impersonating_school"] == "obs-x"

    def test_impersonate_unknown_school_does_not_impersonate(self, admin_client):
        """Impersonating a non-existent school must not start impersonation.

        The route aborts 404; the app's friendly 404 handler turns that into a redirect
        with a Dutch flash, so the observable result is a redirect and an unchanged
        session (no impersonation) — never a silent success.
        """
        resp = admin_client.get("/admin/impersonate/bestaat-niet")
        assert resp.status_code == 302
        with admin_client.session_transaction() as sess:
            assert "impersonating_school" not in sess

    def test_admin_acts_as_school_while_impersonating(self, admin_client):
        """While impersonating, the admin reaches the school's processes page (200).

        Exercises the effective_school_id admin branch returning the impersonated code.
        """
        _create_school(admin_client.application, schoolcode="obs-x")
        admin_client.get("/admin/impersonate/obs-x")
        assert admin_client.get("/processes").status_code == 200

    def test_admin_without_impersonation_is_sent_to_dashboard(self, admin_client):
        """An admin not impersonating anyone is bounced from school routes to the dashboard.

        Exercises the effective_school_id admin branch returning None.
        """
        resp = admin_client.get("/processes")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/admin/")

    def test_stop_impersonating_clears_session_and_returns_to_dashboard(
        self, admin_client
    ):
        """Stopping impersonation removes the session key and returns to the dashboard."""
        _create_school(admin_client.application, schoolcode="obs-x")
        admin_client.get("/admin/impersonate/obs-x")
        resp = admin_client.get("/admin/stop-impersonating")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/admin/")
        with admin_client.session_transaction() as sess:
            assert "impersonating_school" not in sess


class TestAuthAdminPaths:
    """The shared auth helpers handle the admin identity correctly."""

    def test_load_user_resolves_admin_by_prefixed_id(self, unauthed_client):
        """load_user('admin:<id>') returns the Admin (the 'admin:' prefix path)."""
        admin_id = _create_admin(unauthed_client.application)
        with unauthed_client.application.app_context():
            user = load_user(f"admin:{admin_id}")
            assert user is not None
            assert user.is_admin
            assert user.naam == "Beheerder"

    def test_authenticated_admin_redirected_from_school_login(self, admin_client):
        """An authenticated admin hitting the school /login is sent to the admin dashboard."""
        resp = admin_client.get("/login")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/admin/")
