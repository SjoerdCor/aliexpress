"""Auth tests: login wall, credentials, logout, School model, and user_loader.

STAP A + B tests that belong together in one module, separate from the route tests
in test_app.py. Uses the shared client and unauthed_client fixtures from conftest.py.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

from werkzeug.security import check_password_hash, generate_password_hash

from aliexpress.extensions import db
from aliexpress.models import School
from app import app as flask_app
from app import load_user


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
        response = unauthed_client.get("/input_templates/voorkeuren_template.xlsx")
        assert response.status_code != 302
