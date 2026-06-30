"""Auth tests: login wall, credentials, logout, School model, and user_loader.

STAP A + B tests that belong together in one module, separate from the route tests
in test_app.py. Uses the shared client and unauthed_client fixtures from conftest.py.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

from werkzeug.security import check_password_hash, generate_password_hash

from aliexpress.web.extensions import db, limiter
from aliexpress.web.models import School
from aliexpress.web.routes.auth import load_user
from app import app as flask_app


class TestAuthBlueprint:
    """Smoke test: auth blueprint routes are reachable and load_user resolves users."""

    def test_login_route_is_reachable(self, unauthed_client):
        """auth_bp registers /login; route is accessible without authentication."""
        resp = unauthed_client.get("/login")
        assert resp.status_code == 200

    def test_logout_redirects_to_login(self, client):
        """After logout the browser is sent to /login."""
        resp = client.get("/logout")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]


class TestSchoolModel:
    """Tests for the School model: password hashing and Flask-Login user_loader."""

    def _make_school(self, schoolcode="bs-test", naam="Basisschool Test"):
        """Create a School with a known password and return (school, plain_password)."""
        password = "geheim123"
        school = School(
            schoolcode=schoolcode,
            naam=naam,
            password_hash=generate_password_hash(password),
        )
        return school, password

    def test_school_password_hash_is_not_plain_text(self):
        """Stored hash must differ from the plain password."""
        school, password = self._make_school()
        assert school.password_hash != password

    def test_check_password_hash_succeeds_for_correct_password(self):
        """check_password_hash returns True for the original password."""
        school, password = self._make_school()
        assert check_password_hash(school.password_hash, password)

    def test_check_password_hash_fails_for_wrong_password(self):
        """check_password_hash returns False for a different password."""
        school, _ = self._make_school()
        assert not check_password_hash(school.password_hash, "foutWachtwoord!")

    def test_get_id_returns_schoolcode(self):
        """Flask-Login get_id() must return the schoolcode."""
        school, _ = self._make_school(schoolcode="obs-noord")
        assert school.get_id() == "obs-noord"

    def test_user_loader_returns_school_after_persist(self, client):
        """load_user finds a persisted School by schoolcode."""
        school, _ = self._make_school()
        with client.application.app_context():
            db.session.add(school)
            db.session.commit()
            found = load_user("bs-test")
        assert found is not None
        assert found.schoolcode == "bs-test"
        assert found.naam == "Basisschool Test"

    def test_user_loader_returns_none_for_unknown_code(self, client):
        """load_user returns None for a schoolcode that does not exist."""
        with client.application.app_context():
            assert load_user("bestaat-niet") is None


class TestLoginWall:
    """Tests for the login wall: redirect, credentials, logout, and public routes."""

    def _create_school(self, schoolcode="obs-test", password="geheim"):
        """Persist a school and return its schoolcode and plain password."""
        with flask_app.app_context():
            school = School(
                schoolcode=schoolcode,
                naam="Testschool",
                password_hash=generate_password_hash(password),
            )
            db.session.add(school)
            db.session.commit()
        return schoolcode, password

    def test_protected_route_without_session_redirects_to_login(self, unauthed_client):
        """A data route visited without a session redirects to /login."""
        response = unauthed_client.get("/processes")
        assert response.status_code == 302
        assert "/login" in response.headers["Location"]

    def test_login_with_wrong_password_stays_on_login_page(self, unauthed_client):
        """A wrong password re-renders the login page with the Dutch error in the HTML."""
        self._create_school()
        response = unauthed_client.post(
            "/login", data={"schoolcode": "obs-test", "wachtwoord": "fout"}
        )
        assert response.status_code == 200
        assert b"Ongeldige schoolcode of wachtwoord" in response.data

    def test_login_with_wrong_schoolcode_stays_on_login_page(self, unauthed_client):
        """An unknown schoolcode also re-renders login (no enumeration of codes)."""
        response = unauthed_client.post(
            "/login", data={"schoolcode": "bestaat-niet", "wachtwoord": "x"}
        )
        assert response.status_code == 200
        assert b"Ongeldige schoolcode of wachtwoord" in response.data

    def test_login_with_correct_credentials_redirects_to_processes(
        self, unauthed_client
    ):
        """Correct schoolcode and password redirect to /processes."""
        schoolcode, password = self._create_school()
        response = unauthed_client.post(
            "/login", data={"schoolcode": schoolcode, "wachtwoord": password}
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_after_login_protected_route_is_accessible(self, unauthed_client):
        """After logging in, a previously refused data route returns 200."""
        schoolcode, password = self._create_school()
        unauthed_client.post(
            "/login", data={"schoolcode": schoolcode, "wachtwoord": password}
        )
        assert unauthed_client.get("/processes").status_code == 200

    def test_logout_redirects_to_login(self, client):
        """GET /logout redirects to the login page."""
        response = client.get("/logout")
        assert response.status_code == 302
        assert "/login" in response.headers["Location"]

    def test_after_logout_protected_route_is_refused(self, client):
        """After logging out, a data route redirects back to login."""
        client.get("/logout")
        response = client.get("/processes")
        assert response.status_code == 302
        assert "/login" in response.headers["Location"]

    def test_home_is_accessible_without_login(self, unauthed_client):
        """GET / is always public."""
        assert unauthed_client.get("/").status_code == 200

    def test_input_templates_are_accessible_without_login(self, unauthed_client):
        """Template downloads stay public even without a session."""
        response = unauthed_client.get("/input_templates/voorkeuren.xlsx")
        assert response.status_code == 200


class TestLoginRateLimit:
    """The login route is rate-limited against brute-force attempts."""

    def test_repeated_logins_are_eventually_blocked(self, unauthed_client):
        """Past the per-minute cap, attempts get the friendly 429 -> redirect, not 200.

        The fixtures disable the limiter suite-wide (the shared client logs in often); this
        test re-enables it and resets the window so it exercises the limiter in isolation.
        """
        app_under_test = unauthed_client.application
        limiter.enabled = True
        with app_under_test.app_context():
            limiter.reset()
        try:
            statuses = [
                unauthed_client.post(
                    "/login", data={"schoolcode": "x", "wachtwoord": "y"}
                ).status_code
                for _ in range(25)
            ]
        finally:
            limiter.enabled = False
            with app_under_test.app_context():
                limiter.reset()
        assert statuses[0] == 200  # first attempt allowed (wrong-credentials re-render)
        assert (
            302 in statuses
        )  # later attempts blocked -> 429 handler redirects to login

    def test_get_login_is_not_rate_limited(self, unauthed_client):
        """Viewing the login form (GET) is never blocked; only POST attempts count."""
        limiter.enabled = True
        with unauthed_client.application.app_context():
            limiter.reset()
        try:
            statuses = [unauthed_client.get("/login").status_code for _ in range(25)]
        finally:
            limiter.enabled = False
        assert all(status == 200 for status in statuses)


class TestChangePassword:
    """Security tests for the forced-password-change flow.

    A school account flagged must_change_password=True is forced through
    /wachtwoord-instellen before it can reach any other page.  These tests
    guard the invariants that make the flow secure:

    1. The forced redirect actually fires after login.
    2. Accounts that do NOT need a change are kept out (no CSRF bypass).
    3. Weak passwords are rejected and the flag stays True.
    4. Mismatched confirmations are rejected and the flag stays True.
    5. A genuinely strong password completes the flow correctly.
    """

    def _create_school_must_change(
        self,
        app,
        schoolcode="obs-nieuw",
        naam="Nieuwe School",
        password="tijdelijk123",
    ):
        """Persist a school with must_change_password=True; return (schoolcode, password)."""
        with app.app_context():
            school = School(
                schoolcode=schoolcode,
                naam=naam,
                password_hash=generate_password_hash(password),
                must_change_password=True,
            )
            db.session.add(school)
            db.session.commit()
        return schoolcode, password

    def test_login_with_must_change_password_redirects_to_change_password(
        self, unauthed_client
    ):
        """A school flagged must_change_password=True must not reach /processes after
        login — it must be redirected to /wachtwoord-instellen so the admin-set
        temporary password cannot be used indefinitely."""
        schoolcode, password = self._create_school_must_change(
            unauthed_client.application
        )
        response = unauthed_client.post(
            "/login", data={"schoolcode": schoolcode, "wachtwoord": password}
        )
        assert response.status_code == 302
        assert "wachtwoord-instellen" in response.headers["Location"]

    def test_change_password_get_blocked_when_flag_is_false(self, client):
        """A school that does NOT have must_change_password=True hitting
        /wachtwoord-instellen must be redirected away — the page must not be
        accessible to already-configured accounts (prevents accidental re-use)."""
        # The ``client`` fixture logs in as a school with must_change_password=False
        # (the default), so a GET to the route should redirect to /processes.
        response = client.get("/wachtwoord-instellen")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")

    def test_weak_password_rejected_and_flag_stays_true(self, unauthed_client):
        """A trivially weak password (zxcvbn score < 3) must be rejected with a Dutch
        error message and must NOT flip must_change_password to False.  If the flag
        could be cleared by an invalid submission, an attacker who somehow reaches the
        form could lock out the school from ever being forced to change their password.
        """
        schoolcode, password = self._create_school_must_change(
            unauthed_client.application
        )
        unauthed_client.post(
            "/login", data={"schoolcode": schoolcode, "wachtwoord": password}
        )
        response = unauthed_client.post(
            "/wachtwoord-instellen",
            data={"wachtwoord": "abc", "wachtwoord_bevestig": "abc"},
        )
        assert response.status_code == 200
        assert "te makkelijk" in response.data.decode("utf-8")
        with unauthed_client.application.app_context():
            school = db.session.get(School, schoolcode)
            assert school.must_change_password is True

    def test_mismatched_confirmation_rejected_and_flag_stays_true(
        self, unauthed_client
    ):
        """Submitting mismatching wachtwoord / wachtwoord_bevestig fields must
        re-render the form with an error and leave must_change_password True.
        A bypass here would let an attacker set an unknown password."""
        schoolcode, password = self._create_school_must_change(
            unauthed_client.application
        )
        unauthed_client.post(
            "/login", data={"schoolcode": schoolcode, "wachtwoord": password}
        )
        strong = "PaardenBloem!Fiets42Oost"
        response = unauthed_client.post(
            "/wachtwoord-instellen",
            data={"wachtwoord": strong, "wachtwoord_bevestig": strong + "X"},
        )
        assert response.status_code == 200
        assert "niet overeen" in response.data.decode("utf-8")
        with unauthed_client.application.app_context():
            school = db.session.get(School, schoolcode)
            assert school.must_change_password is True

    def test_strong_password_accepted_and_flag_cleared(self, unauthed_client):
        """A strong passphrase (zxcvbn score >= 3, unrelated to schoolcode/naam)
        must update the password hash, flip must_change_password to False, and
        redirect to /processes.  This is the happy path that unlocks the account;
        verifying the hash and the flag ensures no partial writes can leave the
        account in a broken state."""
        schoolcode, old_password = self._create_school_must_change(
            unauthed_client.application
        )
        unauthed_client.post(
            "/login", data={"schoolcode": schoolcode, "wachtwoord": old_password}
        )
        # Passphrase is deliberately unrelated to schoolcode/naam so zxcvbn scores >= 3.
        new_password = "PaardenBloem!Fiets42Oost"
        response = unauthed_client.post(
            "/wachtwoord-instellen",
            data={
                "wachtwoord": new_password,
                "wachtwoord_bevestig": new_password,
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/processes")
        with unauthed_client.application.app_context():
            school = db.session.get(School, schoolcode)
            assert school.must_change_password is False
            assert check_password_hash(school.password_hash, new_password)

    def test_change_password_get_renders_form_for_fresh_account(self, unauthed_client):
        """A fresh school (must_change_password=True) that follows the forced redirect
        must actually be shown the password-setting form — this is the first page the
        teacher sees, so a regression that fails to render it would lock them out."""
        schoolcode, password = self._create_school_must_change(
            unauthed_client.application
        )
        unauthed_client.post(
            "/login", data={"schoolcode": schoolcode, "wachtwoord": password}
        )
        response = unauthed_client.get("/wachtwoord-instellen")
        assert response.status_code == 200
        assert "Wachtwoord instellen" in response.data.decode("utf-8")
