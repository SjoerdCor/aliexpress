"""Auth blueprint: login, logout, change_password routes, and auth helpers."""

import logging

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash
from zxcvbn import zxcvbn

from aliexpress.extensions import db, limiter
from aliexpress.models import Admin, School

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


def effective_school_id():
    """Return the school_id for the current request.

    For a school user: their own schoolcode.
    For an admin impersonating a school: the impersonated school's code.
    For an admin not impersonating: None (caller should redirect to admin dashboard).
    """
    if current_user.is_admin:
        return session.get("impersonating_school")
    return current_user.schoolcode


def load_user(user_id):
    """Return the authenticated user (School or Admin) for the given identity string."""
    if user_id.startswith("admin:"):
        return db.session.get(Admin, int(user_id[6:]))
    return db.session.get(School, user_id)


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("20 per minute", methods=["POST"])
def login():
    """Show and handle the school login form."""
    if current_user.is_authenticated:
        return redirect(
            url_for("admin.dashboard")
            if current_user.is_admin
            else url_for("processes.index")
        )
    if request.method == "POST":
        schoolcode = request.form.get("schoolcode", "").strip()
        password = request.form.get("wachtwoord", "")
        school = db.session.get(School, schoolcode)
        if school is None or not check_password_hash(school.password_hash, password):
            flash("Ongeldige schoolcode of wachtwoord.", "error")
            return render_template("login.html")
        session.clear()
        login_user(school, remember=False)
        if school.must_change_password:
            return redirect(url_for("auth.change_password"))
        return redirect(url_for("processes.index"))
    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    """Log the current user (school or admin) out and redirect to the login page."""
    session.clear()
    logout_user()
    return redirect(url_for("auth.login"))


@auth_bp.route("/wachtwoord-instellen", methods=["GET", "POST"])
@login_required
def change_password():
    """Force a school to set their own password after first login."""
    if current_user.is_admin or not current_user.must_change_password:
        return redirect(url_for("processes.index"))
    if request.method == "POST":
        password = request.form.get("wachtwoord", "")
        confirm = request.form.get("wachtwoord_bevestig", "")
        if password != confirm:
            flash("Wachtwoorden komen niet overeen.", "error")
            return render_template("change_password.html")
        result = zxcvbn(
            password, user_inputs=[current_user.schoolcode, current_user.naam]
        )
        if result["score"] < 3:
            flash(
                "Wachtwoord is te makkelijk te raden. Gebruik meer tekens of vermijd "
                "voor de hand liggende woorden en patronen.",
                "error",
            )
            return render_template("change_password.html")
        current_user.password_hash = generate_password_hash(password)
        current_user.must_change_password = False
        db.session.commit()
        logger.info("School '%s' changed their password", current_user.schoolcode)
        flash("Wachtwoord ingesteld. Je bent nu ingelogd.", "success")
        return redirect(url_for("processes.index"))
    return render_template("change_password.html")
